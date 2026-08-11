#include "inferx/engine/kv_autosize.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace inferx::engine {
namespace {

using ::testing::HasSubstr;

constexpr int64_t kGiB = 1ll << 30;

TEST(KvAutosizeTest, ExplicitBlocksWin) {
  KvSizingSpec spec;
  spec.explicit_blocks = 1234;
  spec.explicit_bytes = 1;  // would be far too small; must be ignored
  spec.block_bytes = 1 << 20;
  auto blocks = ResolveKvBlocks(spec, /*free=*/0, /*total=*/0);
  ASSERT_TRUE(blocks.ok());
  EXPECT_EQ(*blocks, 1234);
}

TEST(KvAutosizeTest, ExplicitBytesBeatUtilization) {
  KvSizingSpec spec;
  spec.explicit_bytes = 4 * kGiB;
  spec.block_bytes = 1 << 20;  // 1 MiB -> 4096 blocks
  spec.gpu_memory_utilization = 0.01;
  auto blocks = ResolveKvBlocks(spec, 1 * kGiB, 80 * kGiB);
  ASSERT_TRUE(blocks.ok());
  EXPECT_EQ(*blocks, 4096);
}

TEST(KvAutosizeTest, UtilizationBudgetArithmetic) {
  KvSizingSpec spec;
  spec.gpu_memory_utilization = 0.5;
  spec.block_bytes = 1 << 20;
  spec.headroom_bytes = 1 * kGiB;
  // total 80 GiB, 10 GiB used by weights: budget = 40 - 10 - 1 = 29 GiB.
  auto blocks = ResolveKvBlocks(spec, 70 * kGiB, 80 * kGiB);
  ASSERT_TRUE(blocks.ok());
  EXPECT_EQ(*blocks, 29 * 1024);
}

TEST(KvAutosizeTest, TooSmallBudgetIsActionable) {
  KvSizingSpec spec;
  spec.gpu_memory_utilization = 0.1;
  spec.block_bytes = 1 << 20;
  spec.min_blocks = 4096;
  // Budget: 8 GiB total*util - 7 GiB used - 1 GiB headroom = 0.
  auto blocks = ResolveKvBlocks(spec, 73 * kGiB, 80 * kGiB);
  ASSERT_FALSE(blocks.ok());
  EXPECT_THAT(std::string(blocks.status().message()),
              HasSubstr("raise --gpu-memory-utilization"));
}

TEST(KvAutosizeTest, ExplicitBytesTooSmallIsActionable) {
  KvSizingSpec spec;
  spec.explicit_bytes = 1 << 20;
  spec.block_bytes = 1 << 20;
  spec.min_blocks = 128;
  auto blocks = ResolveKvBlocks(spec, 0, 0);
  ASSERT_FALSE(blocks.ok());
  EXPECT_THAT(std::string(blocks.status().message()),
              HasSubstr("--kv-cache-memory-bytes"));
}

TEST(KvAutosizeTest, MissingBlockBytesIsAnError) {
  KvSizingSpec spec;
  EXPECT_FALSE(ResolveKvBlocks(spec, kGiB, kGiB).ok());
}

}  // namespace
}  // namespace inferx::engine
