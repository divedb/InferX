#ifndef INFERX_CACHE_CACHE_MANAGER_H_
#define INFERX_CACHE_CACHE_MANAGER_H_

#include <cstdint>
#include <string_view>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/types/span.h"
#include "inferx/cache/kv_block_pool.h"
#include "inferx/cache/prefix_cache.h"

namespace inferx {

/// Owns logical request allocations over borrowed rank-local KV storage.
/// Single-threaded; the pool must outlive this manager. Allocator operations
/// for managed blocks must go through this object. Device execution must
/// finish before Release or destruction: this initial contract has no async leases.
/// Capacity is reserved storage, not a watermark of successfully computed KV.
///
/// With prefix caching enabled, the manager also owns the hash-keyed prefix
/// index: admission adopts and pins the longest cached prompt prefix,
/// completed full blocks are published into the index, released blocks with
/// live entries stay cached (LRU-evictable), and allocation pressure evicts
/// before it fails.
class CacheManager {
 public:
  /// \brief Result of admitting a prompt to the prefix index.
  struct PrefixAdoption {
    /// Whole blocks of prompt KV already computed elsewhere.
    int64_t cached_tokens = 0;
    /// Pinned blocks holding that KV, in logical order. Empty when the
    /// manager runs without prefix caching or nothing matched.
    std::vector<int32_t> block_ids;
  };

  /// \brief Wires the manager to the KV block pool.
  ///
  /// \param pool                 The pool to allocate from; must outlive the
  ///                             manager.
  /// \param enable_prefix_caching Whether completed blocks become shareable
  ///                             prefix-cache entries. When false the manager
  ///                             behaves exactly as before the cache existed.
  explicit CacheManager(KvBlockPool* pool, bool enable_prefix_caching = false);
  ~CacheManager();
  CacheManager(const CacheManager&) = delete;
  CacheManager& operator=(const CacheManager&) = delete;

  /// \brief Adopts the prompt's cached prefix, pinning its blocks.
  ///
  /// Must be called at most once per request, before any Reserve, with the
  /// full prompt. The adoption is capped at `max_cached_tokens` (pass
  /// prompt_len - 1 to keep the last prompt token recomputing for logits)
  /// and truncated to whole blocks. Also records the prompt's hash chain so
  /// later PublishComputed calls extend rather than recompute it. Salt
  /// namespaces the chain: salted and unsalted versions of one prompt share
  /// nothing.
  PrefixAdoption AdoptPrefix(uint64_t request_id, absl::Span<const TokenId> prompt,
                             std::string_view cache_salt, int64_t max_cached_tokens);

  /// \brief Publishes every full block of `prompt + output` beyond what was
  ///        already hashed for this request.
  ///
  /// The two spans concatenated are exactly the tokens whose KV exists —
  /// the accepted output so far, not the token about to be sampled. Calling
  /// this repeatedly is cheap: only newly covered full blocks register.
  /// Block i of the request's table must already hold tokens
  /// [i*block_size, (i+1)*block_size).
  Status PublishComputed(uint64_t request_id, absl::Span<const TokenId> prompt,
                         absl::Span<const TokenId> output);

  /// Ensures capacity for target_tokens and returns only newly granted blocks
  /// in logical order. Growth is all-or-nothing: failure leaves the existing
  /// mapping and pool availability unchanged. A smaller target never shrinks.
  /// With prefix caching, evicts cached blocks before failing.
  StatusOr<std::vector<int32_t>> Reserve(uint64_t request_id, int64_t target_tokens);

  /// Borrowed logical block mapping. Invalidated by growth/release of this
  /// request. An unknown request has an empty mapping.
  absl::Span<const int32_t> Blocks(uint64_t request_id) const;

  /// Releases the request's allocation; unknown/already released IDs are no-ops.
  /// Blocks with live prefix-cache entries stay cached (they become eviction
  /// candidates); the rest return to the pool's free list.
  Status Release(uint64_t request_id);

  /// Drops every prefix-cache entry; evictable blocks return to the pool.
  void ClearPrefixCache();

  /// Entries currently discoverable by hash; zero without prefix caching.
  int64_t cached_prefix_blocks() const;

  /// Whether the prefix index is active for this manager.
  bool prefix_caching_enabled() const { return prefix_ != nullptr; }

  int64_t free_blocks() const { return pool_->FreeBlocks(); }
  int64_t total_blocks() const { return pool_->NumBlocks(); }
  float usage() const;

 private:
  /// Per-request hash-chain state; present only while prefix caching is on.
  struct PrefixState {
    BlockHashChain chain;
    /// Full blocks already hashed (and published where the block was new).
    int64_t hashed_blocks = 0;
  };

  /// Reclaims cached blocks until `extra` blocks are free, or gives up.
  void EvictFor(int64_t extra);

  KvBlockPool* pool_;
  absl::flat_hash_map<uint64_t, std::vector<int32_t>> allocations_;
  absl::flat_hash_map<uint64_t, PrefixState> prefix_state_;
  std::unique_ptr<PrefixCache> prefix_;
};

}  // namespace inferx

#endif  // INFERX_CACHE_CACHE_MANAGER_H_
