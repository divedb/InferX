// Unit tests for the prefix-cache primitives: the SHA-256 block-hash chain
// and the reference-counted, LRU-evictable block index.

#include "inferx/cache/prefix_cache.h"

#include <gtest/gtest.h>

#include <array>
#include <cstring>
#include <string>
#include <vector>

#include "inferx/core/dtype.h"

namespace inferx {
namespace {

constexpr int64_t kBlockSize = 4;

std::vector<BlockHash> HashBlocks(const std::vector<std::vector<TokenId>>& blocks,
                                  std::string_view salt = {}) {
  BlockHashChain chain(salt);
  std::vector<BlockHash> hashes;
  for (const auto& block : blocks) hashes.push_back(chain.Extend(block));
  return hashes;
}

std::string Hex(const BlockHash& digest) {
  static const char* kDigits = "0123456789abcdef";
  std::string out;
  out.reserve(digest.size() * 2);
  for (const uint8_t byte : digest) {
    out.push_back(kDigits[byte >> 4]);
    out.push_back(kDigits[byte & 0xf]);
  }
  return out;
}

std::unique_ptr<KvBlockPool> MakePool(int64_t blocks) {
  KvLayout layout;
  layout.entries_per_token = 2;
  layout.kv_heads = 1;
  layout.head_dim = 8;
  layout.dtype = DataType::kBFloat16;
  return std::make_unique<KvBlockPool>(std::move(
      KvBlockPool::Create(/*num_layers=*/2, blocks, kBlockSize, layout, Device::Cpu()).value()));
}

TEST(Sha256Test, MatchesFipsVectors) {
  // FIPS 180-4 / NIST example vectors.
  EXPECT_EQ(Hex(Sha256::Digest(std::span<const std::byte>())),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
  const std::string abc = "abc";
  EXPECT_EQ(Hex(Sha256::Digest(
                std::span<const std::byte>(reinterpret_cast<const std::byte*>(abc.data()),
                                           abc.size()))),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
  const std::string long_input(1000 * 1000, 'a');
  EXPECT_EQ(Hex(Sha256::Digest(std::span<const std::byte>(
                    reinterpret_cast<const std::byte*>(long_input.data()),
                    long_input.size()))),
            "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0");
}

TEST(Sha256Test, IncrementalMatchesOneShot) {
  const std::string data = "streamed update alignment test payload";
  const auto span = std::span<const std::byte>(
      reinterpret_cast<const std::byte*>(data.data()), data.size());
  Sha256 one_shot;
  one_shot.Update(span);
  Sha256 streamed;
  for (size_t i = 0; i < data.size(); ++i) {
    streamed.Update(span.subspan(i, 1));
  }
  EXPECT_EQ(one_shot.Finish(), streamed.Finish());
  // Finish() resets: a reused instance digests only new input.
  Sha256 reused;
  reused.Update(span);
  reused.Finish();
  EXPECT_EQ(reused.Finish(), Sha256::Digest(std::span<const std::byte>()));
}

TEST(BlockHashChainTest, DeterministicAcrossInstancesAndSalted) {
  const std::vector<std::vector<TokenId>> blocks = {{1, 2, 3, 4}, {5, 6, 7, 8}};
  const auto first = HashBlocks(blocks);
  const auto second = HashBlocks(blocks);
  ASSERT_EQ(first.size(), 2u);
  EXPECT_EQ(first, second);  // Cross-instance determinism, unlike seeded hashes.

  const auto salted = HashBlocks(blocks, "tenant-a");
  EXPECT_NE(first, salted);
  const auto salted_again = HashBlocks(blocks, "tenant-a");
  EXPECT_EQ(salted, salted_again);

  // Any mutation changes the chain from that block onward, never before it.
  auto mutated = blocks;
  mutated[1][2] = 99;
  const auto mutated_hashes = HashBlocks(mutated);
  EXPECT_EQ(mutated_hashes[0], first[0]);
  EXPECT_NE(mutated_hashes[1], first[1]);

  // Block boundaries are encoded: re-splitting the same tokens differently
  // yields different digests.
  const auto resplit = HashBlocks({{1, 2}, {3, 4}, {5, 6, 7, 8}});
  EXPECT_NE(resplit[0], first[0]);
}

TEST(PrefixCacheTest, PublishAdoptReleaseEvictLifecycle) {
  auto pool = MakePool(8);
  PrefixCache cache(pool.get());

  const auto hashes = HashBlocks({{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}});
  const int32_t b0 = pool->AllocateBlock().value();
  const int32_t b1 = pool->AllocateBlock().value();
  cache.Publish(hashes[0], b0);
  cache.Publish(hashes[1], b1);
  ASSERT_EQ(cache.cached_blocks(), 2);
  EXPECT_EQ(cache.evictable_blocks(), 0);  // Publishers hold references.

  // A miss adopts nothing and pins nothing.
  EXPECT_TRUE(cache.Adopt({hashes[2]}).empty());

  // The longest prefix hits; adopted blocks are pinned (not evictable).
  const std::vector<int32_t> adopted = cache.Adopt(hashes);
  EXPECT_EQ(adopted, (std::vector<int32_t>{b0, b1}));
  EXPECT_EQ(cache.evictable_blocks(), 0);  // Publishers still hold references.

  cache.Release({b0, b1});  // Publishers done; entries stay cached.
  cache.Release(adopted);   // Adopter done; now both are evictable.
  EXPECT_EQ(cache.cached_blocks(), 2);
  EXPECT_EQ(cache.evictable_blocks(), 2);
  EXPECT_EQ(pool->FreeBlocks(), 6);  // Bytes retained: nothing freed yet.

  // Eviction reclaims LRU entries back to the pool's free list.
  EXPECT_EQ(cache.Evict(1), 1);
  EXPECT_EQ(pool->FreeBlocks(), 7);
  EXPECT_EQ(cache.cached_blocks(), 1);
  EXPECT_TRUE(cache.Adopt({hashes[0]}).empty());   // Evicted hash is a miss.
  EXPECT_EQ(cache.Adopt({hashes[1]}), (std::vector<int32_t>{b1}));
}

TEST(PrefixCacheTest, PinnedBlocksAreNeverEvictionVictims) {
  auto pool = MakePool(8);
  PrefixCache cache(pool.get());

  const auto hashes = HashBlocks({{1, 2, 3, 4}, {5, 6, 7, 8}});
  const int32_t b0 = pool->AllocateBlock().value();
  const int32_t b1 = pool->AllocateBlock().value();
  cache.Publish(hashes[0], b0);
  cache.Publish(hashes[1], b1);
  cache.Release({b0, b1});
  ASSERT_EQ(cache.evictable_blocks(), 2);

  // Pin the older entry: eviction takes the unpinned one instead.
  const std::vector<int32_t> pinned = cache.Adopt({hashes[0]});
  ASSERT_EQ(pinned, (std::vector<int32_t>{b0}));
  EXPECT_EQ(cache.Evict(5), 1);
  EXPECT_EQ(pool->FreeBlocks(), 7);
  EXPECT_TRUE(cache.Contains(b0));
  EXPECT_FALSE(cache.Contains(b1));

  // Zero evictable blocks left: eviction is a no-op even under pressure.
  EXPECT_EQ(cache.Evict(3), 0);
}

TEST(PrefixCacheTest, ClearWithLiveAdoptionKeepsPinnedBlocksAllocated) {
  auto pool = MakePool(8);
  PrefixCache cache(pool.get());

  const auto hashes = HashBlocks({{1, 2, 3, 4}, {5, 6, 7, 8}});
  const int32_t b0 = pool->AllocateBlock().value();
  const int32_t b1 = pool->AllocateBlock().value();
  cache.Publish(hashes[0], b0);
  cache.Publish(hashes[1], b1);
  cache.Release({b0, b1});  // Publishers done; both evictable.
  ASSERT_EQ(cache.evictable_blocks(), 2);

  // An adoption pins b0; Clear() forgets every entry but must not free the
  // adopted block -- its bytes are still a request's page table.
  const std::vector<int32_t> adopted = cache.Adopt(hashes);
  ASSERT_EQ(adopted, (std::vector<int32_t>{b0, b1}));
  cache.Clear();
  EXPECT_EQ(cache.cached_blocks(), 0);
  EXPECT_EQ(pool->FreeBlocks(), 6);  // Both pinned blocks stay allocated.

  // The adoption's later release finds no entries: the blocks return to
  // the pool through the plain free path, not double-freed here.
  cache.Release(adopted);
  EXPECT_EQ(cache.cached_blocks(), 0);
}

TEST(PrefixCacheTest, DuplicateRegistrationsCoexist) {
  auto pool = MakePool(8);
  PrefixCache cache(pool.get());

  const auto hashes = HashBlocks({{1, 2, 3, 4}});
  const int32_t first = pool->AllocateBlock().value();
  const int32_t second = pool->AllocateBlock().value();
  cache.Publish(hashes[0], first);
  cache.Publish(hashes[0], second);  // Concurrent identical writers.
  EXPECT_EQ(cache.cached_blocks(), 2);

  // Both adopters hit. They may share one physical block (refcounted) or
  // land on the duplicate; either way every adoption is a hit.
  const std::vector<int32_t> one = cache.Adopt(hashes);
  const std::vector<int32_t> another = cache.Adopt(hashes);
  ASSERT_EQ(one.size(), 1u);
  ASSERT_EQ(another.size(), 1u);
  EXPECT_EQ(cache.evictable_blocks(), 0);  // Publishers still hold refs.

  // Every holder releases; both physical blocks become evictable and the
  // index can reclaim both.
  cache.Release(one);
  cache.Release(another);
  cache.Release({first, second});
  EXPECT_EQ(cache.evictable_blocks(), 2);
  EXPECT_EQ(cache.Evict(2), 2);
  EXPECT_EQ(pool->FreeBlocks(), 8);
}

}  // namespace
}  // namespace inferx
