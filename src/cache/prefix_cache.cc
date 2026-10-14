#include "inferx/cache/prefix_cache.h"

#include <cstring>
#include <utility>

namespace inferx {
namespace {

/// Domain separators, versioned so a future encoding change cannot silently
/// mix with chains computed the old way.
constexpr char kChainDomain[] = "inferx.kv.blockhash.v1";
constexpr char kSeedDomain[] = "inferx.kv.blockhash.seed.v1";

void AppendU64(std::vector<std::byte>& out, uint64_t value) {
  for (int i = 0; i < 8; ++i) {
    out.push_back(static_cast<std::byte>(value >> (8 * i)));
  }
}

}  // namespace

BlockHashChain::BlockHashChain(std::string_view cache_salt) {
  // Length-prefixed salt: different lengths with equal bytes stay distinct.
  Sha256 sha;
  sha.Update(kSeedDomain, std::strlen(kSeedDomain));
  std::array<std::byte, 8> length{};
  const uint64_t bytes = static_cast<uint64_t>(cache_salt.size());
  for (int i = 0; i < 8; ++i) length[i] = static_cast<std::byte>(bytes >> (8 * i));
  sha.Update(length.data(), length.size());
  if (!cache_salt.empty()) {
    sha.Update(cache_salt.data(), cache_salt.size());
  }
  tip_ = sha.Finish();
}

BlockHash BlockHashChain::Extend(absl::Span<const TokenId> tokens) {
  std::vector<std::byte> record;
  record.reserve(std::strlen(kChainDomain) + tip_.size() + 8 + tokens.size() * 8);
  const auto append = [&record](const void* data, size_t bytes) {
    const auto* first = static_cast<const std::byte*>(data);
    record.insert(record.end(), first, first + bytes);
  };
  append(kChainDomain, std::strlen(kChainDomain));
  append(tip_.data(), tip_.size());
  AppendU64(record, static_cast<uint64_t>(tokens.size()));
  for (const TokenId token : tokens) {
    AppendU64(record, static_cast<uint64_t>(static_cast<int64_t>(token)));
  }
  tip_ = Sha256::Digest(record);
  return tip_;
}

std::vector<int32_t> PrefixCache::Adopt(absl::Span<const BlockHash> hashes,
                                        int64_t max_blocks) {
  std::vector<int32_t> adopted;
  const int64_t limit =
      max_blocks < 0 ? static_cast<int64_t>(hashes.size())
                     : std::min<int64_t>(max_blocks, static_cast<int64_t>(hashes.size()));
  for (int64_t i = 0; i < limit; ++i) {
    auto it = by_hash_.find(hashes[i]);
    if (it == by_hash_.end() || it->second.empty()) break;
    // First live entry wins, as vLLM's map does.
    Entry* entry = it->second.front();
    if (entry->refs == 0) lru_.erase(entry->lru);
    ++entry->refs;
    adopted.push_back(entry->block);
  }
  return adopted;
}

void PrefixCache::Publish(const BlockHash& hash, int32_t block) {
  if (by_block_.find(block) != by_block_.end()) return;  // One entry per block.
  auto entry = std::make_unique<Entry>();
  entry->hash = hash;
  entry->block = block;
  entry->refs = 1;
  Entry* raw = entry.get();
  by_hash_[hash].push_back(raw);
  by_block_[block] = std::move(entry);
}

void PrefixCache::Release(absl::Span<const int32_t> blocks) {
  for (const int32_t block : blocks) {
    auto it = by_block_.find(block);
    if (it == by_block_.end()) continue;
    Entry* entry = it->second.get();
    if (entry->refs == 0) continue;  // Double release is a caller bug; ignore.
    if (--entry->refs == 0) {
      entry->lru = lru_.insert(lru_.end(), block);
    }
  }
}

void PrefixCache::Unlink(Entry* entry) {
  auto& duplicates = by_hash_[entry->hash];
  for (size_t i = 0; i < duplicates.size(); ++i) {
    if (duplicates[i] == entry) {
      duplicates.erase(duplicates.begin() + static_cast<std::ptrdiff_t>(i));
      break;
    }
  }
  if (duplicates.empty()) by_hash_.erase(entry->hash);
  by_block_.erase(entry->block);
}

int64_t PrefixCache::Evict(int64_t count) {
  int64_t evicted = 0;
  while (evicted < count && !lru_.empty()) {
    const int32_t block = lru_.front();
    lru_.pop_front();
    auto it = by_block_.find(block);
    if (it == by_block_.end()) continue;
    Unlink(it->second.get());
    if (pool_->FreeBlock(block).ok()) ++evicted;
  }
  return evicted;
}

void PrefixCache::Clear() {
  // Free the blocks snapshot first: iterating the owning map while erasing
  // through Unlink would invalidate it.
  std::vector<int32_t> evictable(lru_.begin(), lru_.end());
  for (const int32_t block : evictable) {
    (void)pool_->FreeBlock(block);
  }
  by_hash_.clear();
  by_block_.clear();
  lru_.clear();
}

}  // namespace inferx
