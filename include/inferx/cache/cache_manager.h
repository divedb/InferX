#ifndef INFERX_CACHE_CACHE_MANAGER_H_
#define INFERX_CACHE_CACHE_MANAGER_H_

#include <cstdint>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/types/span.h"
#include "inferx/cache/kv_block_pool.h"

namespace inferx {

/// Owns logical request allocations over borrowed rank-local KV storage.
/// Single-threaded; the pool must outlive this manager. Allocator operations
/// for managed blocks must go through this object. Device execution must finish
/// before Release or destruction: this initial contract has no async leases.
/// Capacity is reserved storage, not a watermark of successfully computed KV.
class CacheManager {
 public:
  explicit CacheManager(KvBlockPool* pool);
  ~CacheManager();
  CacheManager(const CacheManager&) = delete;
  CacheManager& operator=(const CacheManager&) = delete;

  /// Ensures capacity for target_tokens and returns only newly granted blocks
  /// in logical order. Growth is all-or-nothing: failure leaves the existing
  /// mapping and pool availability unchanged. A smaller target never shrinks.
  StatusOr<std::vector<int32_t>> Reserve(uint64_t request_id, int64_t target_tokens);

  /// Borrowed logical block mapping. Invalidated by growth/release of this
  /// request. An unknown request has an empty mapping.
  absl::Span<const int32_t> Blocks(uint64_t request_id) const;

  /// Releases the request's allocation; unknown/already released IDs are no-ops.
  Status Release(uint64_t request_id);

  int64_t free_blocks() const { return pool_->FreeBlocks(); }
  int64_t total_blocks() const { return pool_->NumBlocks(); }
  float usage() const;

 private:
  KvBlockPool* pool_;
  absl::flat_hash_map<uint64_t, std::vector<int32_t>> allocations_;
};

}  // namespace inferx

#endif  // INFERX_CACHE_CACHE_MANAGER_H_
