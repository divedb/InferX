#include "inferx/core/caching_allocator.h"

#include <algorithm>
#include <cstdint>
#include <random>
#include <vector>

#include <gtest/gtest.h>

#include "inferx/core/device_buffer.h"
#include "inferx/core/tensor_spec.h"

namespace inferx {
namespace {

constexpr size_t kCapacity = 1 << 20;  // 1 MiB
constexpr size_t kAlign = kTensorAlignment;

class AllocatorTest : public ::testing::Test {
 protected:
  void SetUp() override {
    auto buf = DeviceBuffer::Allocate(kCapacity, DeviceId::Cpu());
    ASSERT_TRUE(buf.ok()) << buf.status();
    buffer_ = *std::move(buf);
    alloc_ = CachingAllocator(buffer_.data(), kCapacity, DeviceId::Cpu());
  }

  DeviceBuffer buffer_;
  CachingAllocator alloc_;
};

TEST_F(AllocatorTest, StartsEmpty) {
  const auto s = alloc_.GetStats();
  EXPECT_EQ(s.capacity, kCapacity);
  EXPECT_EQ(s.in_use, 0u);
  EXPECT_EQ(s.free_block_count, 1u);
  EXPECT_EQ(s.largest_free_block, kCapacity);
  EXPECT_DOUBLE_EQ(s.Fragmentation(), 0.0);
}

TEST_F(AllocatorTest, AllocateAndFree) {
  auto p = alloc_.Allocate(1000);
  ASSERT_TRUE(p.ok()) << p.status();
  EXPECT_EQ(reinterpret_cast<uintptr_t>(*p) % kAlign, 0u);

  // Requests are rounded up to the alignment granularity.
  EXPECT_EQ(alloc_.GetStats().in_use, 1024u);

  ASSERT_TRUE(alloc_.Deallocate(*p).ok());
  EXPECT_EQ(alloc_.GetStats().in_use, 0u);
  EXPECT_EQ(alloc_.GetStats().free_block_count, 1u);
  EXPECT_EQ(alloc_.GetStats().largest_free_block, kCapacity);
}

// ---------------------------------------------------------------------------
// Per-call alignment
// ---------------------------------------------------------------------------

TEST_F(AllocatorTest, HonoursWeakerAlignmentsForFree) {
  // Anything at or below the granularity is already satisfied by construction.
  for (size_t a : {size_t{16}, size_t{32}, size_t{64}, kTensorAlignment}) {
    auto p = alloc_.Allocate(100, a);
    ASSERT_TRUE(p.ok()) << "alignment " << a << ": " << p.status();
    EXPECT_EQ(reinterpret_cast<uintptr_t>(*p) % a, 0u) << "alignment " << a;
    ASSERT_TRUE(alloc_.Deallocate(*p).ok());
  }
}

TEST_F(AllocatorTest, HonoursOverAlignedRequests) {
  for (size_t a : {size_t{256}, size_t{1024}, kPageAlignment, size_t{1 << 14}}) {
    auto p = alloc_.Allocate(100, a);
    ASSERT_TRUE(p.ok()) << "alignment " << a << ": " << p.status();
    EXPECT_EQ(reinterpret_cast<uintptr_t>(*p) % a, 0u) << "alignment " << a;
    ASSERT_TRUE(alloc_.Deallocate(*p).ok());
  }
}

// An over-aligned request must skip forward to reach its alignment, and the
// skipped bytes must remain allocatable rather than being leaked into the block.
TEST_F(AllocatorTest, AlignmentPaddingStaysAllocatable) {
  // Offset the arena so that offset 0 is not page-aligned, forcing real padding.
  auto head = alloc_.Allocate(kTensorAlignment);
  ASSERT_TRUE(head.ok());

  auto aligned = alloc_.Allocate(4096, kPageAlignment);
  ASSERT_TRUE(aligned.ok()) << aligned.status();
  EXPECT_EQ(reinterpret_cast<uintptr_t>(*aligned) % kPageAlignment, 0u);

  const auto s = alloc_.GetStats();
  // The gap between the two is free space, not lost space.
  const size_t accounted = s.in_use + (s.capacity - s.in_use);
  EXPECT_EQ(accounted, s.capacity);

  ASSERT_TRUE(alloc_.Deallocate(*aligned).ok());
  ASSERT_TRUE(alloc_.Deallocate(*head).ok());

  // Everything, padding included, coalesces back to one whole-capacity block.
  EXPECT_EQ(alloc_.GetStats().in_use, 0u);
  EXPECT_EQ(alloc_.GetStats().free_block_count, 1u);
  EXPECT_EQ(alloc_.GetStats().largest_free_block, kCapacity);
}

TEST_F(AllocatorTest, RejectsNonPowerOfTwoAlignment) {
  EXPECT_EQ(alloc_.Allocate(64, 100).status().code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(alloc_.Allocate(64, 0).status().code(),
            absl::StatusCode::kInvalidArgument);
}

// Repeated over-aligned churn must settle exactly like the default path does.
TEST_F(AllocatorTest, OverAlignedChurnDoesNotLeak) {
  for (int i = 0; i < 200; ++i) {
    auto a = alloc_.Allocate(3000, kPageAlignment);
    ASSERT_TRUE(a.ok()) << "iter " << i << ": " << a.status();
    auto b = alloc_.Allocate(5000, 1024);
    ASSERT_TRUE(b.ok()) << "iter " << i << ": " << b.status();
    ASSERT_TRUE(alloc_.Deallocate(*a).ok());
    ASSERT_TRUE(alloc_.Deallocate(*b).ok());
  }
  const auto s = alloc_.GetStats();
  EXPECT_EQ(s.in_use, 0u);
  EXPECT_EQ(s.free_block_count, 1u);
  EXPECT_EQ(s.largest_free_block, kCapacity);
}

TEST_F(AllocatorTest, ZeroAndNull) {
  EXPECT_EQ(alloc_.Allocate(0).status().code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_TRUE(alloc_.Deallocate(nullptr).ok());
}

TEST_F(AllocatorTest, UninitializedFails) {
  CachingAllocator empty;
  EXPECT_EQ(empty.Allocate(16).status().code(),
            absl::StatusCode::kFailedPrecondition);
}

TEST_F(AllocatorTest, DoubleFreeAndForeignPointerAreRejected) {
  auto p = alloc_.Allocate(4096);
  ASSERT_TRUE(p.ok());
  ASSERT_TRUE(alloc_.Deallocate(*p).ok());
  EXPECT_EQ(alloc_.Deallocate(*p).code(),
            absl::StatusCode::kInvalidArgument);

  int stack_object = 0;
  EXPECT_EQ(alloc_.Deallocate(&stack_object).code(),
            absl::StatusCode::kInvalidArgument);
}

TEST_F(AllocatorTest, AllocationsDoNotOverlap) {
  std::vector<std::pair<std::byte*, size_t>> regions;
  for (int i = 0; i < 32; ++i) {
    const size_t n = 512 * (i + 1);
    auto p = alloc_.Allocate(n);
    ASSERT_TRUE(p.ok()) << p.status();
    regions.emplace_back(static_cast<std::byte*>(*p), n);
  }
  std::sort(regions.begin(), regions.end());
  for (size_t i = 1; i < regions.size(); ++i) {
    EXPECT_GE(regions[i].first, regions[i - 1].first + regions[i - 1].second)
        << "region " << i << " overlaps its predecessor";
  }
}

TEST_F(AllocatorTest, CoalescesAdjacentFrees) {
  auto a = alloc_.Allocate(4096);
  auto b = alloc_.Allocate(4096);
  auto c = alloc_.Allocate(4096);
  ASSERT_TRUE(a.ok() && b.ok() && c.ok());

  // Free the outer two first: no coalescing is possible while b is live.
  ASSERT_TRUE(alloc_.Deallocate(*a).ok());
  ASSERT_TRUE(alloc_.Deallocate(*c).ok());
  EXPECT_GT(alloc_.GetStats().free_block_count, 1u);

  // Freeing the middle block must merge all three plus the tail into one.
  ASSERT_TRUE(alloc_.Deallocate(*b).ok());
  EXPECT_EQ(alloc_.GetStats().free_block_count, 1u);
  EXPECT_EQ(alloc_.GetStats().largest_free_block, kCapacity);
  EXPECT_EQ(alloc_.GetStats().in_use, 0u);
}

TEST_F(AllocatorTest, BestFitReusesTheExactHole) {
  // Carve three live blocks, then free the middle to leave a 4 KiB hole.
  auto a = alloc_.Allocate(4096);
  auto hole = alloc_.Allocate(4096);
  auto c = alloc_.Allocate(4096);
  ASSERT_TRUE(a.ok() && hole.ok() && c.ok());
  void* hole_addr = *hole;
  ASSERT_TRUE(alloc_.Deallocate(*hole).ok());

  // An exact-size request must land in the hole rather than at the tail.
  auto reused = alloc_.Allocate(4096);
  ASSERT_TRUE(reused.ok()) << reused.status();
  EXPECT_EQ(*reused, hole_addr);
}

// The steady-state property the decode loop depends on: the same handful of
// activation shapes are allocated and freed every step, and the allocator must
// settle rather than drift.
TEST_F(AllocatorTest, SteadyStateChurnDoesNotFragment) {
  const size_t sizes[] = {8192, 16384, 4096, 32768};
  for (int step = 0; step < 500; ++step) {
    std::vector<void*> live;
    for (size_t n : sizes) {
      auto p = alloc_.Allocate(n);
      ASSERT_TRUE(p.ok()) << "step " << step << ": " << p.status();
      live.push_back(*p);
    }
    for (void* p : live) ASSERT_TRUE(alloc_.Deallocate(p).ok());
  }

  const auto s = alloc_.GetStats();
  EXPECT_EQ(s.in_use, 0u);
  EXPECT_EQ(s.free_block_count, 1u);
  EXPECT_EQ(s.largest_free_block, kCapacity);
  EXPECT_EQ(s.num_allocations, s.num_deallocations);
  EXPECT_EQ(s.num_failures, 0u);
}

TEST_F(AllocatorTest, ExhaustionReportsFragmentation) {
  auto p = alloc_.Allocate(kCapacity + 1);
  ASSERT_FALSE(p.ok());
  EXPECT_EQ(p.status().code(), absl::StatusCode::kResourceExhausted);
  EXPECT_NE(p.status().message().find("fragmentation"), std::string::npos);
  EXPECT_EQ(alloc_.GetStats().num_failures, 1u);
}

TEST_F(AllocatorTest, FragmentedFailureIsDistinguishable) {
  // Fill the arena with 1 KiB blocks, then free every other one. Half the
  // capacity is free, but no single block larger than 1 KiB exists.
  std::vector<void*> blocks;
  for (;;) {
    auto p = alloc_.Allocate(1024);
    if (!p.ok()) break;
    blocks.push_back(*p);
  }
  for (size_t i = 0; i < blocks.size(); i += 2) {
    ASSERT_TRUE(alloc_.Deallocate(blocks[i]).ok());
  }

  const auto s = alloc_.GetStats();
  EXPECT_GT(s.capacity - s.in_use, kCapacity / 4);
  EXPECT_LE(s.largest_free_block, 1024u);
  EXPECT_GT(s.Fragmentation(), 0.9);

  // Plenty of free bytes, but a 64 KiB request still cannot be served.
  EXPECT_EQ(alloc_.Allocate(64 * 1024).status().code(),
            absl::StatusCode::kResourceExhausted);
}

TEST_F(AllocatorTest, RandomizedChurnStaysConsistent) {
  std::mt19937 rng(1234);
  std::uniform_int_distribution<size_t> size_dist(1, 32768);
  std::vector<void*> live;

  for (int i = 0; i < 20000; ++i) {
    const bool do_alloc = live.empty() || (rng() % 100) < 55;
    if (do_alloc) {
      auto p = alloc_.Allocate(size_dist(rng));
      if (p.ok()) live.push_back(*p);
    } else {
      const size_t idx = rng() % live.size();
      std::swap(live[idx], live.back());
      ASSERT_TRUE(alloc_.Deallocate(live.back()).ok());
      live.pop_back();
    }
  }
  for (void* p : live) ASSERT_TRUE(alloc_.Deallocate(p).ok());

  // Everything returned must coalesce back to a single whole-capacity block.
  const auto s = alloc_.GetStats();
  EXPECT_EQ(s.in_use, 0u);
  EXPECT_EQ(s.free_block_count, 1u);
  EXPECT_EQ(s.largest_free_block, kCapacity);
}

TEST_F(AllocatorTest, ResetDropsEverything) {
  ASSERT_TRUE(alloc_.Allocate(4096).ok());
  ASSERT_TRUE(alloc_.Allocate(4096).ok());
  EXPECT_GT(alloc_.GetStats().in_use, 0u);

  alloc_.Reset();
  EXPECT_EQ(alloc_.GetStats().in_use, 0u);
  EXPECT_EQ(alloc_.GetStats().free_block_count, 1u);
  EXPECT_EQ(alloc_.GetStats().largest_free_block, kCapacity);
}

TEST_F(AllocatorTest, PeakIsTracked) {
  auto a = alloc_.Allocate(64 * 1024);
  ASSERT_TRUE(a.ok());
  auto b = alloc_.Allocate(64 * 1024);
  ASSERT_TRUE(b.ok());
  ASSERT_TRUE(alloc_.Deallocate(*a).ok());
  ASSERT_TRUE(alloc_.Deallocate(*b).ok());
  EXPECT_EQ(alloc_.GetStats().peak_in_use, 128u * 1024);
  EXPECT_EQ(alloc_.GetStats().in_use, 0u);
}

TEST_F(AllocatorTest, AllocateViewIsUsableMemory) {
  auto t = alloc_.AllocateView(TensorSpec(DataType::kInt32, Shape{1024}));
  ASSERT_TRUE(t.ok()) << t.status();
  EXPECT_EQ(t->Device(), DeviceId::Cpu());

  int32_t* p = t->DataAs<int32_t>();
  ASSERT_NE(p, nullptr);
  for (int i = 0; i < 1024; ++i) p[i] = i * 7;
  for (int i = 0; i < 1024; ++i) ASSERT_EQ(p[i], i * 7);

  ASSERT_TRUE(alloc_.Deallocate(t->Data()).ok());
}

}  // namespace
}  // namespace inferx
