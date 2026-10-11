#include "inferx/cache/cache_manager.h"

#include <limits>
#include <memory>
#include <vector>

#include "gtest/gtest.h"

namespace inferx {
namespace {

class CacheManagerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    KvLayout layout;
    layout.kv_heads = 1;
    layout.head_dim = 8;
    auto pool = KvBlockPool::Create(1, 3, 4, layout, Device::Cpu());
    ASSERT_TRUE(pool.ok()) << pool.status();
    pool_ = std::make_unique<KvBlockPool>(std::move(*pool));
  }
  std::unique_ptr<KvBlockPool> pool_;
};

TEST_F(CacheManagerTest, GrowthReturnsOnlyNewBlocksAndKeepsLogicalOrder) {
  CacheManager cache(pool_.get());
  auto first = cache.Reserve(1, 3);
  ASSERT_TRUE(first.ok());
  ASSERT_EQ(first->size(), 1);
  auto next = cache.Reserve(1, 9);
  ASSERT_TRUE(next.ok());
  ASSERT_EQ(next->size(), 2);
  EXPECT_EQ(cache.Blocks(1)[0], first->front());
  EXPECT_EQ(cache.Blocks(1)[1], (*next)[0]);
  EXPECT_EQ(cache.Blocks(1)[2], (*next)[1]);
  auto smaller = cache.Reserve(1, 1);
  ASSERT_TRUE(smaller.ok());
  EXPECT_TRUE(smaller->empty());
  EXPECT_EQ(cache.Blocks(1).size(), 3);
  EXPECT_FLOAT_EQ(cache.usage(), 1.0f);
}

TEST_F(CacheManagerTest, FailedGrowthPreservesMappingAndFreeBlocks) {
  CacheManager cache(pool_.get());
  ASSERT_TRUE(cache.Reserve(1, 4).ok());
  const int32_t original = cache.Blocks(1)[0];
  EXPECT_EQ(cache.Reserve(1, 16).status().code(), absl::StatusCode::kResourceExhausted);
  ASSERT_EQ(cache.Blocks(1).size(), 1);
  EXPECT_EQ(cache.Blocks(1)[0], original);
  EXPECT_EQ(cache.free_blocks(), 2);
  // Another request can still consume every remaining block.
  auto other = cache.Reserve(2, 8);
  ASSERT_TRUE(other.ok());
  EXPECT_EQ(other->size(), 2);
}

TEST_F(CacheManagerTest, FailedAdmissionDoesNotLeavePartialAllocation) {
  CacheManager cache(pool_.get());
  EXPECT_FALSE(cache.Reserve(1, 16).ok());
  EXPECT_TRUE(cache.Blocks(1).empty());
  EXPECT_EQ(cache.free_blocks(), 3);
  ASSERT_TRUE(cache.Release(1).ok());
  EXPECT_EQ(cache.free_blocks(), 3);
}

TEST_F(CacheManagerTest, ReleaseIsIdempotentAndIdsCanBeReused) {
  CacheManager cache(pool_.get());
  ASSERT_TRUE(cache.Reserve(42, 12).ok());
  ASSERT_TRUE(cache.Release(42).ok());
  ASSERT_TRUE(cache.Release(42).ok());
  EXPECT_TRUE(cache.Blocks(42).empty());
  EXPECT_EQ(cache.free_blocks(), 3);
  auto reused = cache.Reserve(42, 12);
  ASSERT_TRUE(reused.ok());
  EXPECT_EQ(reused->size(), 3);
}

TEST_F(CacheManagerTest, DestructorReleasesOutstandingRequests) {
  {
    CacheManager cache(pool_.get());
    ASSERT_TRUE(cache.Reserve(1, 4).ok());
    ASSERT_TRUE(cache.Reserve(2, 8).ok());
    EXPECT_EQ(pool_->FreeBlocks(), 0);
  }
  EXPECT_EQ(pool_->FreeBlocks(), 3);
}

TEST_F(CacheManagerTest, InvalidAndExtremeCapacityDoNotChangeAllocation) {
  CacheManager cache(pool_.get());
  EXPECT_EQ(cache.Reserve(1, -1).status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(cache.Reserve(1, std::numeric_limits<int64_t>::max()).status().code(),
            absl::StatusCode::kResourceExhausted);
  auto empty = cache.Reserve(1, 0);
  ASSERT_TRUE(empty.ok());
  EXPECT_TRUE(empty->empty());
  EXPECT_TRUE(cache.Blocks(1).empty());
  EXPECT_EQ(cache.free_blocks(), 3);
}

}  // namespace
}  // namespace inferx
