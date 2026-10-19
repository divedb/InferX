#ifndef INFERX_CACHE_CACHE_MANAGER_H_
#define INFERX_CACHE_CACHE_MANAGER_H_

#include <cstdint>
#include <memory>
#include <string_view>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/types/span.h"
#include "inferx/cache/block_hash.h"
#include "inferx/cache/cache_group_manager.h"
#include "inferx/cache/cache_spec.h"
#include "inferx/cache/kv_block_pool.h"
#include "inferx/cache/recurrent_state_pool.h"
#include "inferx/core/status.h"
#include "inferx/tokenizer/id.h"  // TokenId

namespace inferx {

/// \brief Request-level facade over the cache groups.
///
/// One CacheGroupManager knows one group. This class owns what spans groups:
///  - per-request state, one RequestGroupState per planned group;
///  - the request's block-hash chain (one chain suffices, because every
///    prefix-index key already carries its group index and phase 1 uses a
///    uniform block size);
///  - prefix adoption: the usable hit is the longest length every token-wise
///    group can supply, found by calling FindLongestCacheHit with shrinking
///    caps until the groups agree;
///  - atomic multi-group Reserve and Fork, with rollback;
///  - request-level prefix statistics and the device work queued for the
///    runner.
///
/// Counterpart of vLLM's KVCacheManager plus KVCacheCoordinator.
///
/// The pools are borrowed and must outlive this object; the plan is copied.
/// Not thread-safe: call from the scheduler thread only.
class CacheManager {
 public:
  /// \brief Result of admitting a prompt to the prefix index.
  struct PrefixAdoption {
    /// Agreed whole-block cached tokens: the longest length every token-wise
    /// group can supply, not simply the minimum of per-group hits.
    int64_t cached_tokens = 0;

    /// Block ids per token-wise group, in logical order, indexed by group
    /// index. A window or chunk group leads with kNullBlockId for the part of
    /// the prefix that lies before its retention floor. Empty when prefix
    /// caching is off or nothing hit.
    std::vector<std::vector<BlockId>> group_block_ids;
  };

  /// \brief Prefix-caching counters, per request and in tokens. Pool-level
  /// counters, when added, count per block probe; keep the scopes distinct.
  struct Stats {
    int64_t prefix_queries = 0;
    int64_t prefix_hits = 0;
    int64_t prefix_queried_tokens = 0;
    int64_t prefix_hit_tokens = 0;
  };

  /// \brief Device work the runner must perform before this step's kernels
  /// run. Drained by TakeRunnerWork.
  struct RunnerWork {
    /// State slots newly assigned to requests; the runner zeroes them with
    /// RecurrentStatePool::ResetSlot.
    std::vector<int64_t> slots_to_reset;
    /// Recurrent-state copies for forked children.
    std::vector<SlotCopy> slot_copies;
    /// Block-page copies for copy-on-write after a fork.
    std::vector<BlockCopy> block_copies;
  };

  /// \brief Builds one manager per planned group.
  ///
  /// \param pool        Block pool for the token-wise groups (null only when
  ///                    the plan has none).
  /// \param state_pool  State pool for the recurrent group (null only when the
  ///                    plan has none).
  /// \param plan        From PlanCacheGroups; copied.
  /// \return InvalidArgument when a needed pool is null, the plan's block size
  ///         differs from the pool's, or a group manager cannot be created.
  static StatusOr<std::unique_ptr<CacheManager>> Create(KvBlockPool* pool,
                                                        RecurrentStatePool* state_pool,
                                                        const CachePlan& plan,
                                                        bool enable_prefix_caching);

  ~CacheManager();
  CacheManager(const CacheManager&) = delete;
  CacheManager& operator=(const CacheManager&) = delete;

  // -- admission and growth ---------------------------------------------------

  /// \brief Registers `request_id` and adopts the longest cached prefix of
  /// `prompt`. At most once per request, before any Reserve. The hit is
  /// capped by `max_cached_tokens` (the caller's n-1 rule), truncated to whole
  /// blocks, and pinned. `cache_salt` namespaces the hash chain, so tenants
  /// never share prefixes. With caching off the request is still registered
  /// and the adoption is empty.
  ///
  /// While the plan has a recurrent group the agreed length is 0: KV blocks
  /// cannot be reused without the matching recurrent state, and state boundary
  /// sharing does not exist yet. Counted as a miss.
  ///
  /// \return InvalidArgument for a duplicate request id; other errors leave no
  ///         pin and no registered request.
  StatusOr<PrefixAdoption> AdoptPrefix(uint64_t request_id, absl::Span<const TokenId> prompt,
                                       std::string_view cache_salt, int64_t max_cached_tokens);

  /// \brief Grows `request_id`'s holdings in every group to `target`;
  /// all-or-nothing across groups. Order per call: roll off tokens below each
  /// group's retention floor, check capacity, reserve each group, and roll
  /// back the groups already reserved if a later one fails. Never shrinks. A
  /// request not yet registered is registered on its first Reserve.
  ///
  /// Rolled-off blocks stay released even when the call then fails: they are
  /// dead for this request either way.
  ///
  /// \return ResourceExhausted only when eviction cannot help (no free block
  ///         or state slot); the request is unchanged apart from roll-off.
  Status Reserve(uint64_t request_id, const GrowthTarget& target);

  /// \brief Whether Reserve(request_id, target) would succeed now. Does not
  /// change anything. Conservative: it does not count blocks that roll-off
  /// would free. An unknown request is costed as a fresh one.
  bool CanReserve(uint64_t request_id, const GrowthTarget& target) const;

  /// \brief Publishes every newly completed full block, per token-wise group,
  /// to the prefix index. `num_computed_tokens` bounds what is complete: the
  /// hash chain is extended over `prompt` then `output` as far as full blocks
  /// within `num_computed_tokens` reach. Publishes nothing while the plan has a
  /// recurrent group (the entries could never be reused). No-op with caching
  /// off.
  Status PublishComputed(uint64_t request_id, absl::Span<const TokenId> prompt,
                         absl::Span<const TokenId> output, int64_t num_computed_tokens);

  /// \brief Forks a live request into children (beam-style branching).
  /// Children share token-wise blocks (ref-counted; copy-on-write when a
  /// child writes a shared partial block) and get fresh recurrent slots with
  /// copied state; the copies are queued as RunnerWork. Children must be
  /// unregistered ids; the parent stays live. All-or-nothing across groups and
  /// children. Each child should subsequently go through PublishComputed with
  /// its own token stream.
  Status Fork(uint64_t parent_request_id, absl::Span<const uint64_t> child_request_ids);

  /// \brief Releases every holding of `request_id`, tail blocks first. Unknown
  /// or already-released ids are no-ops.
  Status Release(uint64_t request_id);

  /// \brief Drains the device work queued by Reserve and Fork.
  RunnerWork TakeRunnerWork();

  // -- prefix index -----------------------------------------------------------

  /// \brief Drops every prefix-index entry. Fails (FailedPrecondition) while
  /// any request holds blocks, as the pool does: there are no tombstones, so
  /// drain requests first.
  Status ResetPrefixCache();

  // -- per-request views -------------------------------------------------------

  /// \brief Number of token-wise groups; the valid range for `group_index`
  /// below. Hybrid callers branch on this.
  int64_t NumTokenWiseGroups() const;

  /// \brief True when the plan has a recurrent group.
  bool HasRecurrentGroup() const;

  /// \brief One token-wise group's block table for a request, in logical
  /// order, with kNullBlockId for rolled-off entries. Empty for an unknown
  /// request. Invalidated by Reserve, Fork and Release.
  absl::Span<const BlockId> GroupBlocks(uint64_t request_id, int64_t group_index) const;

  /// \brief One group's skipped tokens at `num_computed_tokens`: what the
  /// runner turns into window_left per layer. 0 for full-attention groups.
  int64_t GroupSkippedTokens(int64_t group_index, int64_t num_computed_tokens) const;

  /// \brief The state slot of a request, or -1 if it has none (or the plan has
  /// no recurrent group).
  int64_t StateSlot(uint64_t request_id) const;

  // -- counters ----------------------------------------------------------------

  bool PrefixCachingEnabled() const;
  int64_t BlockSize() const;
  /// \brief Blocks available to Reserve: free plus evictable.
  int64_t NumFreeBlocks() const;
  int64_t NumBlocks() const;
  /// \brief Free state slots; 0 when the plan has no recurrent group.
  int64_t NumFreeSlots() const;
  double Usage() const;
  int64_t NumCachedBlocks() const;
  const Stats& stats() const { return stats_; }
  void ResetStats() { stats_ = Stats{}; }

 private:
  /// Everything the facade keeps per live request.
  struct RequestState {
    /// One per planned group, in plan order (token-wise first, recurrent last).
    std::vector<RequestGroupState> groups;

    /// The request's block-hash chain, one digest per hashed full block, and
    /// the cache salt it is rooted in.
    std::vector<BlockHash> hashes;
    std::vector<uint8_t> salt;
  };

  CacheManager(KvBlockPool* pool, RecurrentStatePool* state_pool, CachePlan plan,
               bool enable_prefix_caching);

  /// Longest whole-block length every token-wise group can supply, found by
  /// shrinking the cap until each group's FindLongestCacheHit agrees.
  CacheHit FindAgreedHit(absl::Span<const BlockHash> hashes, int64_t max_tokens,
                         std::vector<CacheHit>* per_group) const;

  /// Extends `request`'s chain to cover `num_blocks` full blocks of
  /// prompt followed by output.
  void ExtendHashes(RequestState& request, absl::Span<const TokenId> prompt,
                    absl::Span<const TokenId> output, int64_t num_blocks) const;

  Status ReleaseRequest(RequestState& request);

  KvBlockPool* const pool_;
  RecurrentStatePool* const state_pool_;
  const CachePlan plan_;
  const bool enable_prefix_caching_;

  /// One manager per planned group, in plan order.
  std::vector<CacheGroupManagerPtr> managers_;
  absl::flat_hash_map<uint64_t, RequestState> requests_;

  RunnerWork pending_work_;
  Stats stats_;
};

}  // namespace inferx

#endif  // INFERX_CACHE_CACHE_MANAGER_H_