#include "inferx/cache/kv_block_pool.h"

#include <cstdint>
#include <vector>

#include "gtest/gtest.h"

namespace inferx {
namespace {

constexpr int64_t kBlockSize = 16;

CacheComponent PerToken(int64_t bytes_per_token) {
  return CacheComponent::PerToken(bytes_per_token, DataType::kBFloat16);
}

BlockKey Key(int seed, int32_t group = 0) {
  BlockKey key;
  key.hash.bytes[0] = static_cast<uint8_t>(seed);
  key.hash.bytes[1] = static_cast<uint8_t>(seed >> 8);
  key.group_index = group;
  return key;
}

int64_t ByteOffset(const Tensor& from, const Tensor& to) {
  return static_cast<int64_t>(reinterpret_cast<const uint8_t*>(to.Bytes()) -
                              reinterpret_cast<const uint8_t*>(from.Bytes()));
}

TEST(KvBlockPoolTest, CreateRejectsNonPositiveDimensions) {
  const CacheComponent component = PerToken(64);
  EXPECT_EQ(KvBlockPool::Create(0, 4, kBlockSize, component, Device::Cpu()).status().code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(KvBlockPool::Create(2, 0, kBlockSize, component, Device::Cpu()).status().code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(KvBlockPool::Create(2, 4, 0, component, Device::Cpu()).status().code(),
            absl::StatusCode::kInvalidArgument);
}

TEST(KvBlockPoolTest, CreateRejectsFixedSlotAndDegenerateComponents) {
  // Fixed-slot state pages nothing; it belongs to the recurrent state pool.
  const CacheComponent fixed = CacheComponent::FixedSlot(1024, DataType::kFloat32);
  EXPECT_EQ(KvBlockPool::Create(2, 4, kBlockSize, fixed, Device::Cpu()).status().code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(KvBlockPool::Create(2, 4, kBlockSize, PerToken(0), Device::Cpu()).status().code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(
      KvBlockPool::Create(4, kBlockSize, std::vector<CacheComponent>{}, Device::Cpu())
          .status()
          .code(),
      absl::StatusCode::kInvalidArgument);
}

TEST(KvBlockPoolTest, FreshPoolHandsOutUsableBlocksInAscendingOrder) {
  auto pool = KvBlockPool::Create(2, 4, kBlockSize, PerToken(8), Device::Cpu());
  ASSERT_TRUE(pool.ok()) << pool.status();
  // Id 0 is the reserved null block; usable ids run 1..num_blocks.
  for (int32_t expected = 1; expected <= 4; ++expected) {
    auto block = pool->AllocateBlock();
    ASSERT_TRUE(block.ok()) << block.status();
    EXPECT_EQ(*block, expected);
  }
  EXPECT_EQ(pool->NumLayers(), 2);
  EXPECT_EQ(pool->NumBlocks(), 4);
  EXPECT_EQ(pool->BlockSize(), kBlockSize);
  EXPECT_EQ(pool->FreeBlocks(), 0);
  EXPECT_EQ(pool->UsedBlocks(), 4);
  // 2 layers x (4 usable + 1 null) rows x 16 tokens x 8 bytes.
  EXPECT_EQ(pool->Bytes(), 1280u);
}

TEST(KvBlockPoolTest, ExhaustionReturnsResourceExhausted) {
  auto pool = KvBlockPool::Create(1, 2, kBlockSize, PerToken(8), Device::Cpu());
  ASSERT_TRUE(pool.ok()) << pool.status();
  ASSERT_TRUE(pool->AllocateBlock().ok());
  ASSERT_TRUE(pool->AllocateBlock().ok());
  EXPECT_EQ(pool->AllocateBlock().status().code(), absl::StatusCode::kResourceExhausted);
  // The failed allocation leaves the accounting untouched.
  EXPECT_EQ(pool->FreeBlocks(), 0);
  EXPECT_EQ(pool->UsedBlocks(), 2);
}

TEST(KvBlockPoolTest, ReleaseReturnsBlockToLifoFront) {
  auto pool = KvBlockPool::Create(1, 4, kBlockSize, PerToken(8), Device::Cpu());
  ASSERT_TRUE(pool.ok()) << pool.status();
  ASSERT_TRUE(pool->AllocateBlock().ok());   // 1
  auto second = pool->AllocateBlock();       // 2
  ASSERT_TRUE(second.ok());
  ASSERT_TRUE(pool->ReleaseBlock(*second).ok());
  // LIFO: the most recently released block is handed back out first.
  auto reused = pool->AllocateBlock();
  ASSERT_TRUE(reused.ok());
  EXPECT_EQ(*reused, *second);
  EXPECT_EQ(pool->RefCount(*reused), 1u);
  EXPECT_EQ(pool->UsedBlocks(), 2);
}

TEST(KvBlockPoolTest, ReleaseWithoutReferenceIsRejected) {
  auto pool = KvBlockPool::Create(1, 2, kBlockSize, PerToken(8), Device::Cpu());
  ASSERT_TRUE(pool.ok()) << pool.status();
  auto block = pool->AllocateBlock();
  ASSERT_TRUE(block.ok());

  EXPECT_EQ(pool->ReleaseBlock(KvBlockPool::kNullBlockId).code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(pool->ReleaseBlock(-1).code(), absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(pool->ReleaseBlock(3).code(), absl::StatusCode::kInvalidArgument);
  ASSERT_TRUE(pool->ReleaseBlock(*block).ok());
  // Double release: the block holds no reference anymore.
  EXPECT_EQ(pool->ReleaseBlock(*block).code(), absl::StatusCode::kInvalidArgument);

  auto other = pool->AllocateBlock();
  ASSERT_TRUE(other.ok());
  // A batch stops at the first invalid entry.
  EXPECT_EQ(pool->ReleaseBlocks({*other, 99}).code(), absl::StatusCode::kInvalidArgument);
}

TEST(KvBlockPoolTest, AcquireSharesBlocksThroughReferenceCounts) {
  auto pool = KvBlockPool::Create(1, 2, kBlockSize, PerToken(8), Device::Cpu());
  ASSERT_TRUE(pool.ok()) << pool.status();
  auto block = pool->AllocateBlock();
  ASSERT_TRUE(block.ok());

  ASSERT_TRUE(pool->AcquireBlock(*block).ok());
  EXPECT_EQ(pool->RefCount(*block), 2u);
  EXPECT_EQ(pool->UsedBlocks(), 1);  // still one block handed out

  // The null block, unknown ids, and free blocks cannot be acquired.
  EXPECT_EQ(pool->AcquireBlock(KvBlockPool::kNullBlockId).code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(pool->AcquireBlock(3).code(), absl::StatusCode::kInvalidArgument);
  auto other = pool->AllocateBlock();
  ASSERT_TRUE(other.ok());
  ASSERT_TRUE(pool->ReleaseBlock(*other).ok());
  EXPECT_EQ(pool->AcquireBlock(*other).code(), absl::StatusCode::kInvalidArgument);

  ASSERT_TRUE(pool->ReleaseBlock(*block).ok());
  EXPECT_EQ(pool->RefCount(*block), 1u);
  ASSERT_TRUE(pool->ReleaseBlock(*block).ok());
  EXPECT_EQ(pool->RefCount(*block), 0u);
  EXPECT_EQ(pool->FreeBlocks(), 2);
}

TEST(KvBlockPoolTest, AllocateBlocksIsAllOrNothing) {
  auto pool = KvBlockPool::Create(1, 3, kBlockSize, PerToken(8), Device::Cpu());
  ASSERT_TRUE(pool.ok()) << pool.status();

  auto three = pool->AllocateBlocks(3);
  ASSERT_TRUE(three.ok()) << three.status();
  EXPECT_EQ(three->size(), 3u);
  EXPECT_EQ(pool->FreeBlocks(), 0);

  auto none = pool->AllocateBlocks(1);
  EXPECT_EQ(none.status().code(), absl::StatusCode::kResourceExhausted);
  EXPECT_EQ(pool->FreeBlocks(), 0);
}

TEST(KvBlockPoolTest, PublishedBlocksAreDiscoverableAndEvictable) {
  auto pool = KvBlockPool::Create(1, 2, kBlockSize, PerToken(8), Device::Cpu());
  ASSERT_TRUE(pool.ok()) << pool.status();
  auto block = pool->AllocateBlock();
  ASSERT_TRUE(block.ok());

  pool->PublishBlock(Key(1), *block);
  EXPECT_EQ(pool->LookupCached(Key(1)), std::optional<BlockId>(*block));
  EXPECT_EQ(pool->LookupCached(Key(2)), std::nullopt);
  EXPECT_TRUE(pool->Contains(*block));
  EXPECT_EQ(pool->NumCachedBlocks(), 1);
  EXPECT_EQ(pool->EvictableBlocks(), 0);  // still pinned by its holder

  // Releasing the last reference keeps the entry: the block joins the LRU
  // rather than the free queue (the pool's other block was never allocated).
  ASSERT_TRUE(pool->ReleaseBlock(*block).ok());
  EXPECT_EQ(pool->FreeBlocks(), 1);
  EXPECT_EQ(pool->EvictableBlocks(), 1);
  EXPECT_EQ(pool->LookupCached(Key(1)), std::optional<BlockId>(*block));

  // A key is namespaced by its group.
  EXPECT_EQ(pool->LookupCached(Key(1, /*group=*/7)), std::nullopt);
}

TEST(KvBlockPoolTest, AllocationEvictsLruCachedBlocksUnderPressure) {
  auto pool = KvBlockPool::Create(1, 2, kBlockSize, PerToken(8), Device::Cpu());
  ASSERT_TRUE(pool.ok()) << pool.status();
  auto first = pool->AllocateBlock();
  auto second = pool->AllocateBlock();
  ASSERT_TRUE(first.ok() && second.ok());

  pool->PublishBlock(Key(1), *first);
  pool->PublishBlock(Key(2), *second);
  ASSERT_TRUE(pool->ReleaseBlock(*first).ok());
  ASSERT_TRUE(pool->ReleaseBlock(*second).ok());

  // Least recently released is evicted first.
  auto fresh = pool->AllocateBlock();
  ASSERT_TRUE(fresh.ok());
  EXPECT_EQ(*fresh, *first);
  EXPECT_FALSE(pool->Contains(*first));
  EXPECT_EQ(pool->LookupCached(Key(1)), std::nullopt);
  EXPECT_EQ(pool->NumCachedBlocks(), 1);
  EXPECT_EQ(pool->EvictCached(1), 1);  // the survivor
  EXPECT_EQ(pool->EvictableBlocks(), 0);
  // One usable block is still held by `fresh`; only the evicted one frees.
  EXPECT_EQ(pool->FreeBlocks(), 1);
}

TEST(KvBlockPoolTest, ClearCachedEntriesReclaimsEvictableAndTombstonesPinned) {
  auto pool = KvBlockPool::Create(1, 2, kBlockSize, PerToken(8), Device::Cpu());
  ASSERT_TRUE(pool.ok()) << pool.status();
  auto pinned = pool->AllocateBlock();
  auto idle = pool->AllocateBlock();
  ASSERT_TRUE(pinned.ok() && idle.ok());

  pool->PublishBlock(Key(1), *pinned);
  pool->PublishBlock(Key(2), *idle);
  ASSERT_TRUE(pool->ReleaseBlock(*idle).ok());

  pool->ClearCachedEntries();
  EXPECT_EQ(pool->NumCachedBlocks(), 0);
  EXPECT_EQ(pool->LookupCached(Key(1)), std::nullopt);
  EXPECT_EQ(pool->FreeBlocks(), 1);  // the idle one was reclaimed
  EXPECT_TRUE(pool->Contains(*pinned));  // tombstone while still held

  // The pinned tombstone frees normally once its holder releases.
  ASSERT_TRUE(pool->ReleaseBlock(*pinned).ok());
  EXPECT_EQ(pool->FreeBlocks(), 2);
  EXPECT_FALSE(pool->Contains(*pinned));
}

TEST(KvBlockPoolTest, UniformPoolExposesDisjointLayerRegions) {
  auto pool = KvBlockPool::Create(2, 3, 4, PerToken(8), Device::Cpu());
  ASSERT_TRUE(pool.ok()) << pool.status();

  auto first = pool->LayerCache(0);
  ASSERT_TRUE(first.ok()) << first.status();
  EXPECT_EQ(first->GetDataType(), DataType::kUInt8);
  // One row per block id, the null block's included.
  EXPECT_EQ(first->Dims(), (std::vector<int64_t>({4, 4, 8})));
  EXPECT_EQ(first->Numel(), 128);

  auto second = pool->LayerCache(1);
  ASSERT_TRUE(second.ok()) << second.status();
  // Regions are adjacent and exactly one layer apart: 4 rows x 4 tokens x
  // 8 bytes.
  EXPECT_EQ(ByteOffset(*first, *second), 128);

  EXPECT_EQ(pool->LayerCache(-1).status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(pool->LayerCache(2).status().code(), absl::StatusCode::kInvalidArgument);

  const CacheComponent& component = pool->Component();
  EXPECT_EQ(component.bytes_per_token, 8);
  EXPECT_EQ(pool->ComponentFor(1).bytes_per_token, 8);
}

TEST(KvBlockPoolTest, HeterogeneousPoolSizesEachLayerFromItsComponent) {
  auto pool = KvBlockPool::Create(2, kBlockSize, {PerToken(16), PerToken(48)},
                                  Device::Cpu());
  ASSERT_TRUE(pool.ok()) << pool.status();

  // 3 rows x 16 tokens x (16 + 48) bytes.
  EXPECT_EQ(pool->Bytes(), 3072u);
  EXPECT_EQ(pool->NumLayers(), 2);

  auto first = pool->LayerCache(0);
  ASSERT_TRUE(first.ok()) << first.status();
  EXPECT_EQ(first->Dims(), (std::vector<int64_t>({3, 16, 16})));
  auto second = pool->LayerCache(1);
  ASSERT_TRUE(second.ok()) << second.status();
  EXPECT_EQ(second->Dims(), (std::vector<int64_t>({3, 16, 48})));
  // Layer 1 starts after layer 0's whole region.
  EXPECT_EQ(ByteOffset(*first, *second), 768);

  EXPECT_EQ(pool->Component().bytes_per_token, 16);
  EXPECT_EQ(pool->ComponentFor(0).bytes_per_token, 16);
  EXPECT_EQ(pool->ComponentFor(1).bytes_per_token, 48);
}

TEST(KvBlockPoolTest, BlockBytesMatchesRegionMath) {
  EXPECT_EQ(KvBlockPool::BlockBytes(3, kBlockSize, PerToken(64)), 3 * kBlockSize * 64);
}

TEST(KvBlockPoolTest, BlocksForTokensCeilDivides) {
  auto pool = KvBlockPool::Create(1, 8, kBlockSize, PerToken(8), Device::Cpu());
  ASSERT_TRUE(pool.ok()) << pool.status();
  EXPECT_EQ(pool->BlocksForTokens(0), 0);
  EXPECT_EQ(pool->BlocksForTokens(1), 1);
  EXPECT_EQ(pool->BlocksForTokens(kBlockSize), 1);
  EXPECT_EQ(pool->BlocksForTokens(kBlockSize + 1), 2);
  EXPECT_EQ(pool->BlocksForTokens(3 * kBlockSize + 1), 4);
}

TEST(KvBlockPoolTest, EvictCachedNeverTakesPinnedBlocks) {
  auto pool = KvBlockPool::Create(1, 2, kBlockSize, PerToken(8), Device::Cpu());
  ASSERT_TRUE(pool.ok()) << pool.status();
  auto first = pool->AllocateBlock();
  auto second = pool->AllocateBlock();
  ASSERT_TRUE(first.ok() && second.ok());
  pool->PublishBlock(Key(1), *first);
  pool->PublishBlock(Key(2), *second);
  ASSERT_TRUE(pool->ReleaseBlock(*first).ok());
  ASSERT_TRUE(pool->ReleaseBlock(*second).ok());
  ASSERT_EQ(pool->EvictableBlocks(), 2);

  // Pin the older entry: eviction takes the unpinned one instead.
  ASSERT_TRUE(pool->AcquireBlock(*first).ok());
  EXPECT_EQ(pool->EvictCached(5), 1);
  EXPECT_EQ(pool->NumCachedBlocks(), 1);
  EXPECT_EQ(pool->LookupCached(Key(1)), std::optional<BlockId>(*first));
  EXPECT_FALSE(pool->Contains(*second));

  // Nothing evictable left: eviction is a no-op even under pressure.
  EXPECT_EQ(pool->EvictCached(3), 0);
}

TEST(KvBlockPoolTest, RepublishingAKeyRebindsIt) {
  auto pool = KvBlockPool::Create(1, 3, kBlockSize, PerToken(8), Device::Cpu());
  ASSERT_TRUE(pool.ok()) << pool.status();
  const BlockKey key = Key(1);
  auto first = pool->AllocateBlock();
  auto second = pool->AllocateBlock();
  ASSERT_TRUE(first.ok() && second.ok());

  pool->PublishBlock(key, *first);
  pool->PublishBlock(key, *second);  // concurrent writer of an identical prefix
  EXPECT_EQ(pool->LookupCached(key), std::optional<BlockId>(*second));
  // One discoverable entry; the displaced block keeps only its holder's
  // reference until released.
  EXPECT_EQ(pool->NumCachedBlocks(), 1);
  EXPECT_FALSE(pool->Contains(*first));
  ASSERT_TRUE(pool->ReleaseBlock(*first).ok());
  EXPECT_EQ(pool->FreeBlocks(), 2);  // 3 usable, `second` still held

  // The binding survives the displaced block's release.
  ASSERT_TRUE(pool->ReleaseBlock(*second).ok());
  EXPECT_EQ(pool->EvictableBlocks(), 1);
  EXPECT_EQ(pool->LookupCached(key), std::optional<BlockId>(*second));
}

TEST(BlockTableTest, LocateResolvesPositionToBlockAndSlot) {
  BlockTable table(4);
  table.Append(10);
  table.Append(11);
  table.Append(12);

  BlockId block = -1;
  int64_t slot = -1;
  ASSERT_TRUE(table.Locate(0, &block, &slot));
  EXPECT_EQ(block, 10);
  EXPECT_EQ(slot, 0);
  ASSERT_TRUE(table.Locate(3, &block, &slot));
  EXPECT_EQ(block, 10);
  EXPECT_EQ(slot, 3);
  ASSERT_TRUE(table.Locate(4, &block, &slot));
  EXPECT_EQ(block, 11);
  EXPECT_EQ(slot, 0);
  ASSERT_TRUE(table.Locate(11, &block, &slot));
  EXPECT_EQ(block, 12);
  EXPECT_EQ(slot, 3);

  // Past the allocated blocks, or before the sequence, there is nothing.
  EXPECT_FALSE(table.Locate(12, &block, &slot));
  EXPECT_FALSE(table.Locate(-1, &block, &slot));
}

TEST(BlockTableTest, SizeCapacityAndClear) {
  BlockTable table(4);
  table.Append(7);
  table.Append(8);
  EXPECT_EQ(table.Blocks(), (std::vector<BlockId>({7, 8})));
  EXPECT_EQ(table.Size(), 2);
  EXPECT_EQ(table.CapacityTokens(), 8);

  table.Clear();
  EXPECT_EQ(table.Size(), 0);
  EXPECT_EQ(table.CapacityTokens(), 0);
  BlockId block = -1;
  int64_t slot = -1;
  EXPECT_FALSE(table.Locate(0, &block, &slot));
}

}  // namespace
}  // namespace inferx
