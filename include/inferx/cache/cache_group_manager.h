#ifndef INFERX_CACHE_CACHE_GROUP_MANAGER_H_
#define INFERX_CACHE_CACHE_GROUP_MANAGER_H_

#include <cstdint>
#include <memory>
#include <optional>
#include <variant>
#include <vector>

#include "absl/types/span.h"
#include "inferx/cache/block_hash.h"
#include "inferx/cache/cache_spec.h"
#include "inferx/cache/kv_block_pool.h"
#include "inferx/cache/recurrent_state_pool.h"
#include "inferx/core/status.h"

namespace inferx {

/// The reserved block id used for "no block" in tables and hits.
constexpr BlockId kNullBlockId = KvBlockPool::kNullBlockId;

// ---------------------------------------------------------------------------
// Per-request state
// ---------------------------------------------------------------------------

/// \brief A request's holdings in a token-wise group.
struct TokenWiseRequestState {
  /// Block table in logical order, one entry per block of the sequence.
  /// Positions below the retention floor (rolled off a window, or before the
  /// window on a cache hit) hold kNullBlockId so the table stays indexable by
  /// logical block.
  std::vector<BlockId> blocks;

  /// Tokens adopted from the prefix cache at admission (whole blocks).
  int64_t num_cached_tokens = 0;

  /// Leading blocks already handled by PublishCompleted (the publish cursor).
  int64_t num_published_blocks = 0;
};

/// \brief A request's holdings in the recurrent group.
struct RecurrentRequestState {
  /// Assigned state slot; -1 while unallocated.
  int64_t slot = -1;
};

/// \brief Per-request, per-group bookkeeping owned by a group manager. The
/// alternative matches the group's kind, so a recurrent state can never carry
/// a block table.
using RequestGroupState = std::variant<TokenWiseRequestState, RecurrentRequestState>;

/// \brief Where a request is and where one step takes it.
struct GrowthTarget {
  /// Tokens already computed. Drives window and chunk roll-off.
  int64_t num_computed_tokens = 0;
  /// Tokens the request holds after this step.
  int64_t target_tokens = 0;
  /// Encoder length, for cross-attention groups; 0 elsewhere.
  int64_t encoder_tokens = 0;
};

// ---------------------------------------------------------------------------
// Operation results
// ---------------------------------------------------------------------------

/// \brief A prefix-cache hit found by a read-only search. Nothing is pinned.
struct CacheHit {
  /// Blocks in logical order. A window group leads with kNullBlockId for the
  /// part of the prefix that lies before its window; sink blocks, when the
  /// group has them, are real blocks.
  std::vector<BlockId> blocks;
  /// Tokens covered; a multiple of the block size.
  int64_t num_tokens = 0;
};

/// \brief One block duplication the runner must perform on device
/// (copy-on-write after a fork).
struct BlockCopy {
  BlockId src = kNullBlockId;
  BlockId dst = kNullBlockId;
};

/// \brief One recurrent-state device copy the runner must perform for a fork.
/// Both the recurrent manager and the facade hand these to the runner.
struct SlotCopy {
  int64_t parent_slot = -1;
  int64_t child_slot = -1;
};

/// \brief What a successful Reserve handed to the request.
struct Granted {
  /// Newly allocated blocks, in logical order, already appended to the
  /// request's table. Null padding is not listed.
  std::vector<BlockId> blocks;
  /// Newly assigned recurrent slot. The runner must ResetSlot it before the
  /// step reads it.
  std::optional<int64_t> slot;
  /// Copy-on-write duplications to perform before the step writes.
  std::vector<BlockCopy> copies;
};

// ---------------------------------------------------------------------------
// Manager
// ---------------------------------------------------------------------------

/// \brief Lifecycle policy for one cache group. Implementations are `final`
/// and created only by CreateCacheGroupManager. Single-threaded with the
/// scheduler; methods run per request per step, never per token.
///
/// Hash-chain extension is facade-owned (the facade holds the per-group
/// chains and passes digests in). Managers own retention, hit search,
/// allocation, publication, and release of blocks and slots.
///
/// Per-step call order for one request, across all of its groups:
///   1. ReleaseSkipped  - free blocks that rolled off the retention window.
///   2. NumNewBlocksNeeded on every group, summed against pool capacity.
///   3. Reserve on every group; if one fails, Rollback the ones that
///      succeeded.
///   4. PublishCompleted once the step's blocks are computed.
class CacheGroupManager {
 public:
  virtual ~CacheGroupManager() = default;

  /// \name Identity
  ///\{
  virtual const CacheGroupSpec& group() const = 0;
  int64_t group_index() const { return group().group_index; }
  bool IsTokenWise() const { return group().spec.IsTokenWise(); }
  ///\}

  /// \brief Longest usable cached prefix of at most `max_tokens` tokens, as
  /// whole blocks. Read-only: takes no references, so a coordinator can call
  /// it repeatedly with shrinking caps until every group's hit agrees (a hybrid
  /// model's usable hit is the longest length all groups can supply).
  /// `hashes` are the prompt's full-block digests for this group;
  /// `max_tokens` already includes the facade's n-1 rule. Recurrent groups
  /// return an empty hit in phase E.
  virtual CacheHit FindLongestCacheHit(absl::Span<const BlockHash> hashes,
                                       int64_t max_tokens) const = 0;

  /// \brief Blocks (token-wise) or slots (recurrent: always 1) this group
  /// holds once the request reaches `target_tokens`. The retention cap: cdiv
  /// for full attention, the window formula for sliding window, the encoder
  /// length's blocks for cross-attention.
  virtual int64_t NumBlocksHeldAt(int64_t target_tokens, int64_t encoder_tokens = 0) const = 0;

  /// \brief Installs a hit into the request state and pins its blocks.
  /// All-or-nothing: on failure no pin remains. Sets `num_cached_tokens`.
  virtual Status AdoptCached(RequestGroupState& req, const CacheHit& hit) = 0;

  /// \brief Frees blocks that rolled off the retention floor at
  /// `num_computed_tokens`, rewriting their table entries to kNullBlockId.
  /// Window and chunk groups only; a no-op elsewhere. Kept separate from
  /// Reserve so the size query and Reserve both see the post-roll state.
  virtual Status ReleaseSkipped(RequestGroupState& /*req*/, int64_t /*num_computed_tokens*/) {
    return OkStatus();
  }

  /// \brief New blocks (or slots) Reserve would need right now, given what
  /// the request already holds and the roll-off already applied. When
  /// `pending_hit` is set, cached blocks in it that currently sit idle in the
  /// free queue count as consumed capacity, because pinning them removes them
  /// from the free count.
  virtual int64_t NumNewBlocksNeeded(const RequestGroupState& req, const GrowthTarget& target,
                                     const CacheHit* pending_hit = nullptr) const = 0;

  /// \brief Grows the request's holdings to cover `target`; all-or-nothing.
  /// Pads positions below the retention floor with kNullBlockId so the table
  /// stays logically indexed. On failure the request state is unchanged.
  virtual StatusOr<Granted> Reserve(RequestGroupState& req, const GrowthTarget& target) = 0;

  /// \brief Undoes a successful Reserve: releases what `granted` handed out
  /// and removes it from the request. Lets the facade make a multi-group
  /// reserve atomic when a later group fails.
  virtual Status Rollback(RequestGroupState& req, const Granted& granted) = 0;

  /// \brief Publishes newly completed blocks to the prefix index, starting at
  /// the request's publish cursor. `hashes` are this group's per-block
  /// digests, covering at least every full block up to `num_computed_tokens`.
  /// Skips null (rolled-off) entries. A recurrent group publishes only at
  /// state boundaries (phase E: no-op); an encoder group never publishes.
  virtual Status PublishCompleted(RequestGroupState& req, absl::Span<const BlockHash> hashes,
                                  int64_t num_computed_tokens) = 0;

  /// \brief Releases the request's holdings and resets its state. Blocks are
  /// released tail first, so the free queue evicts the tail before shared
  /// prefixes. Blocks with other references stay put; cached blocks stay
  /// discoverable until evicted; slots return to the state pool.
  virtual Status Release(RequestGroupState& req) = 0;

  /// \brief Forks `parent` into `num_children` children. Token-wise: children
  /// share the parent's blocks through AcquireBlock, and the first Reserve
  /// that writes a shared partial block duplicates it (reported as a
  /// BlockCopy in Granted). Recurrent: children get fresh slots, with one
  /// SlotCopy per child appended to `slot_copies` for the runner to perform.
  virtual Status Fork(const RequestGroupState& parent, int64_t num_children,
                      std::vector<RequestGroupState>* children,
                      std::vector<SlotCopy>* slot_copies) = 0;

  /// \brief Tokens below the retention floor this group skips at
  /// `num_computed_tokens` (window groups: the window formula floored at
  /// sink_tokens; everyone else: 0). Feeds window_left.
  virtual int64_t NumSkippedTokens(int64_t /*num_computed_tokens*/) const { return 0; }
};

using CacheGroupManagerPtr = std::unique_ptr<CacheGroupManager>;

/// \brief The only way to obtain a manager. Dispatches on the retention
/// alternative: FullRetention -> FullAttentionManager, WindowRetention ->
/// SlidingWindowManager, RecurrentRetention -> RecurrentStateManager.
///
/// Both pools are borrowed and must outlive the manager. `state_pool` may be
/// null for token-wise groups; `pool` may be null for the recurrent group.
///
/// Errors: InvalidArgument when a required pool is null or the spec does not
/// validate against the pool's block size.
StatusOr<CacheGroupManagerPtr> CreateCacheGroupManager(const CacheGroupSpec& group,
                                                       KvBlockPool* pool,
                                                       RecurrentStatePool* state_pool);

}  // namespace inferx

#endif  // INFERX_CACHE_CACHE_GROUP_MANAGER_H_