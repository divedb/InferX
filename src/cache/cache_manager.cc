#include "inferx/cache/cache_manager.h"

#include <cassert>

namespace inferx {

CacheManager::CacheManager(KvBlockPool* pool) : pool_(pool) { assert(pool_ != nullptr); }

CacheManager::~CacheManager() {
  for (const auto& entry : allocations_) {
    const Status status = pool_->FreeBlocks(entry.second);
    assert(status.ok());
    (void)status;
  }
}

StatusOr<std::vector<int32_t>> CacheManager::Reserve(uint64_t request_id,
                                                     int64_t target_tokens) {
  if (target_tokens < 0) return InvalidArgumentError("negative cache capacity");
  // Division without addition avoids overflowing for very large token counts.
  const int64_t needed =
      target_tokens / pool_->BlockSize() + (target_tokens % pool_->BlockSize() != 0);
  auto it = allocations_.find(request_id);
  const int64_t held = it == allocations_.end() ? 0 : it->second.size();
  if (needed <= held) return std::vector<int32_t>{};
  const int64_t extra = needed - held;
  if (extra > pool_->FreeBlocks()) {
    return ResourceExhaustedError("insufficient KV blocks for request ", request_id);
  }

  std::vector<int32_t> granted;
  granted.reserve(extra);
  for (int64_t i = 0; i < extra; ++i) {
    auto block = pool_->AllocateBlock();
    if (!block.ok()) {
      // Reverse release also restores the pool's LIFO allocation order.
      for (auto b = granted.rbegin(); b != granted.rend(); ++b) {
        const Status status = pool_->FreeBlock(*b);
        assert(status.ok());
        (void)status;
      }
      return block.status();
    }
    granted.push_back(*block);
  }
  auto& blocks = allocations_[request_id];
  blocks.insert(blocks.end(), granted.begin(), granted.end());
  return granted;
}

absl::Span<const int32_t> CacheManager::Blocks(uint64_t request_id) const {
  auto it = allocations_.find(request_id);
  if (it == allocations_.end()) return {};
  return absl::MakeConstSpan(it->second);
}

Status CacheManager::Release(uint64_t request_id) {
  auto it = allocations_.find(request_id);
  if (it == allocations_.end()) return OkStatus();
  INFERX_RETURN_IF_ERROR(pool_->FreeBlocks(it->second));
  allocations_.erase(it);
  return OkStatus();
}

float CacheManager::usage() const {
  if (pool_->NumBlocks() == 0) return 0.0f;
  return static_cast<float>(pool_->UsedBlocks()) / static_cast<float>(pool_->NumBlocks());
}

}  // namespace inferx
