#include "inferx/cache/cache_manager.h"

#include <cassert>
#include <utility>

namespace inferx {

CacheManager::CacheManager(KvBlockPool* pool, bool enable_prefix_caching)
    : pool_(pool) {
  assert(pool_ != nullptr);
  if (enable_prefix_caching) prefix_ = std::make_unique<PrefixCache>(pool_);
}

CacheManager::~CacheManager() {
  for (const auto& entry : allocations_) {
    if (prefix_ != nullptr) prefix_->Release(entry.second);
    const Status status = pool_->FreeBlocks(entry.second);
    assert(status.ok());
    (void)status;
  }
}

CacheManager::PrefixAdoption CacheManager::AdoptPrefix(uint64_t request_id,
                                                        absl::Span<const TokenId> prompt,
                                                        std::string_view cache_salt,
                                                        int64_t max_cached_tokens) {
  PrefixAdoption adoption;
  if (prefix_ == nullptr || prompt.empty() || allocations_.contains(request_id)) {
    return adoption;
  }

  const int64_t block_size = pool_->BlockSize();
  const int64_t full_blocks = static_cast<int64_t>(prompt.size()) / block_size;
  if (full_blocks == 0) return adoption;

  // Candidate hashes come from a scratch chain; the stored state keeps a
  // fresh chain so PublishComputed walks the identical sequence from the
  // seed (publishing only blocks that do not already have entries).
  BlockHashChain candidates(cache_salt);
  std::vector<BlockHash> hashes;
  hashes.reserve(static_cast<size_t>(full_blocks));
  for (int64_t b = 0; b < full_blocks; ++b) {
    hashes.push_back(candidates.Extend(
        prompt.subspan(static_cast<size_t>(b * block_size), static_cast<size_t>(block_size))));
  }

  // The logits rule: never hand out the very last prompt token. Truncation
  // to whole blocks happens through the block cap.
  const int64_t cap_blocks =
      max_cached_tokens < 0 ? full_blocks : max_cached_tokens / block_size;
  adoption.block_ids = prefix_->Adopt(hashes, cap_blocks);
  adoption.cached_tokens = static_cast<int64_t>(adoption.block_ids.size()) * block_size;

  PrefixState state;
  state.chain = BlockHashChain(cache_salt);
  state.hashed_blocks = 0;
  allocations_[request_id] = adoption.block_ids;
  prefix_state_[request_id] = std::move(state);
  return adoption;
}

Status CacheManager::PublishComputed(uint64_t request_id,
                                     absl::Span<const TokenId> prompt,
                                     absl::Span<const TokenId> output) {
  if (prefix_ == nullptr) return OkStatus();
  auto state_it = prefix_state_.find(request_id);
  if (state_it == prefix_state_.end()) return OkStatus();
  auto blocks_it = allocations_.find(request_id);
  if (blocks_it == allocations_.end()) return OkStatus();

  const int64_t block_size = pool_->BlockSize();
  const int64_t full_blocks =
      (static_cast<int64_t>(prompt.size()) + static_cast<int64_t>(output.size())) / block_size;
  const std::vector<int32_t>& blocks = blocks_it->second;

  // A block spans prompt, output, or the junction; token j of the
  // concatenated sequence lives in prompt[j] or output[j - prompt.size()].
  const auto token_at = [&](int64_t j) -> TokenId {
    return j < static_cast<int64_t>(prompt.size())
               ? prompt[static_cast<size_t>(j)]
               : output[static_cast<size_t>(j - static_cast<int64_t>(prompt.size()))];
  };

  PrefixState& state = state_it->second;
  std::vector<TokenId> window;
  window.reserve(static_cast<size_t>(block_size));
  for (int64_t b = state.hashed_blocks; b < full_blocks; ++b) {
    if (b >= static_cast<int64_t>(blocks.size())) break;
    window.clear();
    for (int64_t t = b * block_size; t < (b + 1) * block_size; ++t) {
      window.push_back(token_at(t));
    }
    const BlockHash hash = state.chain.Extend(absl::MakeConstSpan(window));
    prefix_->Publish(hash, blocks[static_cast<size_t>(b)]);
  }
  state.hashed_blocks = std::min<int64_t>(full_blocks, static_cast<int64_t>(blocks.size()));
  return OkStatus();
}

void CacheManager::EvictFor(int64_t extra) {
  if (prefix_ == nullptr) return;
  const int64_t shortfall = extra - pool_->FreeBlocks();
  if (shortfall > 0) prefix_->Evict(shortfall);
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
  EvictFor(extra);
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
  prefix_state_.erase(request_id);
  if (prefix_ != nullptr) {
    // Cached blocks keep their bytes and hash entries; they leave through
    // eviction, not the free list. Blocks without entries (partial tails,
    // or entries dropped by Clear) return immediately.
    prefix_->Release(it->second);
    std::vector<int32_t> returning;
    returning.reserve(it->second.size());
    for (const int32_t block : it->second) {
      if (!prefix_->Contains(block)) returning.push_back(block);
    }
    INFERX_RETURN_IF_ERROR(pool_->FreeBlocks(returning));
  } else {
    INFERX_RETURN_IF_ERROR(pool_->FreeBlocks(it->second));
  }
  allocations_.erase(it);
  return OkStatus();
}

void CacheManager::ClearPrefixCache() {
  if (prefix_ != nullptr) prefix_->Clear();
}

int64_t CacheManager::cached_prefix_blocks() const {
  return prefix_ == nullptr ? 0 : prefix_->cached_blocks();
}

float CacheManager::usage() const {
  if (pool_->NumBlocks() == 0) return 0.0f;
  return static_cast<float>(pool_->UsedBlocks()) / static_cast<float>(pool_->NumBlocks());
}

}  // namespace inferx
