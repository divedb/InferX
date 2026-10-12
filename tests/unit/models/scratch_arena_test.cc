// ScratchArena behavior: measured growth, correct reuse (no overlap), and
// a frozen phase whose addresses repeat a deterministic sequence. Patterns
// written through one allocation must survive untouched in its neighbors.
#include <cstdint>
#include <cstring>
#include <random>
#include <utility>
#include <vector>

#include "gtest/gtest.h"
#include "inferx/core/device_runtime.h"
#include "inferx/models/scratch_arena.h"
#include "inferx/ops/flash_attention.h"

namespace inferx {
namespace models {
namespace {

constexpr Device kDevice = Device::Cuda(0);

class ScratchArenaTest : public ::testing::Test {
 protected:
  void SetUp() override {
    auto runtime = RuntimeFor(kDevice);
    ASSERT_TRUE(runtime.ok()) << runtime.status();
    runtime_ = *runtime;
    ASSERT_TRUE(runtime_->Activate().ok());
  }

  DeviceRuntime* runtime_ = nullptr;
};

TEST_F(ScratchArenaTest, ReuseNeverOverlaps) {
  ScratchArena arena = ScratchArena::Measuring(kDevice);

  // A(1M) B(2M) C(1M); free A; D(1M) must land somewhere disjoint from B/C.
  auto* a = *arena.Allocate(1u << 20, kTensorAlignment);
  auto* b = *arena.Allocate(2u << 20, kTensorAlignment);
  auto* c = *arena.Allocate(1u << 20, kTensorAlignment);
  ASSERT_TRUE(arena.Deallocate(a).ok());
  auto* d = *arena.Allocate(1u << 20, kTensorAlignment);

  const auto write = [&](void* ptr, size_t bytes, uint8_t value) {
    std::vector<uint8_t> pattern(bytes, value);
    EXPECT_TRUE(runtime_
                    ->Copy(ptr, pattern.data(), bytes, CopyKind::kHostToDevice)
                    .ok());
  };
  const auto verify = [&](void* ptr, size_t bytes, uint8_t value) {
    std::vector<uint8_t> read(bytes, 0);
    EXPECT_TRUE(runtime_
                    ->Copy(read.data(), ptr, bytes, CopyKind::kDeviceToHost)
                    .ok());
    for (size_t i = 0; i < bytes; ++i) {
      ASSERT_EQ(read[i], value) << "byte " << i << " of a neighbor changed";
    }
  };

  write(b, 2 << 20, 0xB1);
  write(c, 1 << 20, 0xC2);
  write(d, 1 << 20, 0xD3);
  verify(b, 2 << 20, 0xB1);
  verify(c, 1 << 20, 0xC2);
  verify(d, 1 << 20, 0xD3);
}

TEST_F(ScratchArenaTest, StressPatternsHold) {
  ScratchArena arena = ScratchArena::Measuring(kDevice);
  struct Live {
    void* ptr;
    size_t bytes;
    uint8_t value;
  };
  std::mt19937 rng(1234);
  std::vector<Live> live;
  for (int round = 0; round < 200; ++round) {
    if (!live.empty() && rng() % 3 == 0) {
      const size_t index = rng() % live.size();
      ASSERT_TRUE(arena.Deallocate(live[index].ptr).ok());
      live.erase(live.begin() + static_cast<long>(index));
      continue;
    }
    const size_t bytes = 1024 * (1 + rng() % 512);
    auto ptr = arena.Allocate(bytes, kTensorAlignment);
    ASSERT_TRUE(ptr.ok()) << ptr.status();
    if (*ptr == nullptr) continue;
    const uint8_t value = 1 + static_cast<uint8_t>(round % 255);
    std::vector<uint8_t> pattern(bytes, value);
    ASSERT_TRUE(
        runtime_->Copy(*ptr, pattern.data(), bytes, CopyKind::kHostToDevice).ok());
    live.push_back({*ptr, bytes, value});
  }
  // Every live block still holds exactly its own pattern: any overlap or
  // bookkeeping slip shows up as a foreign byte.
  for (const Live& block : live) {
    std::vector<uint8_t> read(block.bytes, 0);
    ASSERT_TRUE(runtime_
                    ->Copy(read.data(), block.ptr, block.bytes,
                           CopyKind::kDeviceToHost)
                    .ok());
    for (size_t i = 0; i < block.bytes; ++i) {
      ASSERT_EQ(read[i], block.value) << "foreign byte in a live block";
    }
  }
}

TEST_F(ScratchArenaTest, FreezeThenServeDeterministically) {
  std::vector<void*> first_run;
  int64_t high_water = 0;
  {
    ScratchArena arena = ScratchArena::Measuring(kDevice);
    for (int i = 0; i < 8; ++i) {
      auto* p = *arena.Allocate(1u << 20, kTensorAlignment);
      ASSERT_TRUE(arena.Deallocate(p).ok());
    }
    high_water = arena.high_water_bytes();
    ASSERT_GT(high_water, 0);
    ASSERT_TRUE(arena.Freeze(kDevice, 0.5).ok());
    // Freeze replaces the backing allocation. Address stability is only
    // promised between executions using the same frozen backing.
    for (int i = 0; i < 8; ++i) {
      auto* p = *arena.Allocate(1u << 20, kTensorAlignment);
      first_run.push_back(p);
      ASSERT_TRUE(arena.Deallocate(p).ok());
    }
    for (int i = 0; i < 8; ++i) {
      auto* p = *arena.Allocate(1u << 20, kTensorAlignment);
      ASSERT_EQ(p, first_run[i]);
      ASSERT_TRUE(arena.Deallocate(p).ok());
    }
  }
  EXPECT_GT(high_water, 0);
}

TEST_F(ScratchArenaTest, AttentionPlanSurvivesBlockListGrowth) {
  ScratchArena arena = ScratchArena::Measuring(kDevice);
  const auto exercise = [&]() {
    // The first carve splits a one-entry block vector, forcing growth.
    auto plan = Tensor::Empty(DataType::kInt32, Shape({13}), &arena);
    ASSERT_TRUE(plan.ok()) << plan.status();
    std::vector<Tensor> neighbors;
    for (int i = 0; i < 64; ++i) {
      auto tensor = Tensor::Empty(DataType::kInt32, Shape({32}), &arena);
      ASSERT_TRUE(tensor.ok()) << tensor.status();
      neighbors.push_back(std::move(*tensor));
    }
    auto qo = Tensor::Empty(DataType::kInt32, Shape({2}), kDevice);
    ASSERT_TRUE(qo.ok()) << qo.status();
    const int32_t offsets[] = {0, 128};
    ASSERT_TRUE(runtime_->Copy(qo->Data(), offsets, sizeof(offsets),
                               CopyKind::kHostToDevice).ok());
    ops::OpContext ctx(*runtime_, DeviceStream{});
    ASSERT_TRUE(ops::PrepareFlashAttention(ctx, *qo, *plan, 2, 4, 64).ok());
    ASSERT_TRUE(runtime_->SynchronizeStream(DeviceStream{}).ok());
    std::vector<int32_t> actual(13);
    ASSERT_TRUE(runtime_->Copy(actual.data(), plan->Data(), plan->NBytes(),
                               CopyKind::kDeviceToHost).ok());
    const std::vector<int32_t> expected{
        0, 0, 0, 0, 0, 1, 2, 3, 0, 0, 0, 0, 0x3fffffff};
    EXPECT_EQ(actual, expected);
  };
  exercise();
  ASSERT_FALSE(HasFatalFailure());
  ASSERT_TRUE(arena.Freeze(kDevice, 0.25).ok());
  exercise();
}

}  // namespace
}  // namespace models
}  // namespace inferx
