#include "inferx/core/device_buffer.h"

#include <cstring>
#include <utility>

#include <gtest/gtest.h>

#include "inferx/core/cuda_utils.h"

namespace inferx {
namespace {

TEST(DeviceBuffer, DefaultIsInvalid) {
  DeviceBuffer b;
  EXPECT_FALSE(b.valid());
  EXPECT_EQ(b.data(), nullptr);
  EXPECT_EQ(b.size(), 0u);
}

TEST(DeviceBuffer, HostAllocationIsAlignedAndPadded) {
  auto b = DeviceBuffer::Allocate(1000, DeviceId::Cpu());
  ASSERT_TRUE(b.ok()) << b.status();
  EXPECT_TRUE(b->valid());
  EXPECT_EQ(b->size(), 1024u);  // rounded up to kTensorAlignment
  EXPECT_EQ(reinterpret_cast<uintptr_t>(b->data()) % kTensorAlignment, 0u);
  EXPECT_EQ(b->device(), DeviceId::Cpu());

  std::memset(b->data(), 0xAB, b->size());
  EXPECT_EQ(static_cast<unsigned char>(b->data()[0]), 0xABu);
  EXPECT_EQ(static_cast<unsigned char>(b->data()[b->size() - 1]), 0xABu);
}

// A DeviceBuffer backs BumpArena and CachingAllocator, both of which hand out
// kTensorAlignment-aligned pointers. If the backing buffer were more weakly
// aligned, they would skip bytes to realign, shifting every offset and silently
// shrinking usable capacity. Pinned here because that failure is invisible
// except as off-by-a-few-bytes capacity assertions elsewhere.
TEST(DeviceBuffer, BackingIsAlignedForSubAllocators) {
  for (size_t n : {1u, 100u, 1024u, 100000u}) {
    auto b = DeviceBuffer::Allocate(n, DeviceId::Cpu());
    ASSERT_TRUE(b.ok()) << b.status();
    EXPECT_EQ(reinterpret_cast<uintptr_t>(b->data()) % kTensorAlignment, 0u)
        << "size " << n;
    EXPECT_GE(b->size(), n);
    EXPECT_EQ(b->size() % kTensorAlignment, 0u) << "size " << n;
  }
}

// The floor that must never be crossed: CUTLASS selects kernel variants on
// pointer alignment, and below 16 B it drops to slower tiles.
TEST(DeviceBuffer, TensorAlignmentClearsTheVectorLoadFloor) {
  static_assert(kTensorAlignment >= 16,
                "float4/int4 vectorized loads require 16 B");
  static_assert(kTensorAlignment % 16 == 0);
  SUCCEED();
}

TEST(DeviceBuffer, ZeroSizeIsValidAndEmpty) {
  auto b = DeviceBuffer::Allocate(0, DeviceId::Cpu());
  ASSERT_TRUE(b.ok()) << b.status();
  EXPECT_FALSE(b->valid());
  EXPECT_EQ(b->size(), 0u);
}

TEST(DeviceBuffer, MoveTransfersOwnership) {
  auto a = DeviceBuffer::Allocate(4096, DeviceId::Cpu());
  ASSERT_TRUE(a.ok());
  std::byte* raw = a->data();

  DeviceBuffer moved = *std::move(a);
  EXPECT_EQ(moved.data(), raw);
  EXPECT_EQ(moved.size(), 4096u);

  DeviceBuffer assigned;
  assigned = std::move(moved);
  EXPECT_EQ(assigned.data(), raw);
  EXPECT_FALSE(moved.valid());  // NOLINT(bugprone-use-after-move)
}

TEST(DeviceBuffer, ResetIsIdempotent) {
  auto b = DeviceBuffer::Allocate(4096, DeviceId::Cpu());
  ASSERT_TRUE(b.ok());
  b->Reset();
  EXPECT_FALSE(b->valid());
  b->Reset();
  EXPECT_FALSE(b->valid());
}

// Asking for device memory in a host-only build must fail cleanly rather than
// returning a host pointer that a kernel would later dereference.
TEST(DeviceBuffer, CudaRequestWithoutSupportFailsClearly) {
  if (kCudaEnabled) GTEST_SKIP() << "built with CUDA support";
  auto b = DeviceBuffer::Allocate(4096, DeviceId::Cuda(0));
  ASSERT_FALSE(b.ok());
  EXPECT_EQ(b.status().code(), absl::StatusCode::kFailedPrecondition);
}

TEST(DeviceBuffer, CudaAllocation) {
  if (!CudaAvailable()) GTEST_SKIP() << "no CUDA device available";

  auto b = DeviceBuffer::Allocate(1 << 20, DeviceId::Cuda(0));
  ASSERT_TRUE(b.ok()) << b.status();
  EXPECT_TRUE(b->valid());
  EXPECT_EQ(b->size(), 1u << 20);
  EXPECT_TRUE(b->device().IsCuda());
  EXPECT_EQ(reinterpret_cast<uintptr_t>(b->data()) % kTensorAlignment, 0u);
}

#ifdef INFERX_WITH_CUDA
// Round-trips real bytes through device memory. This is the first test that
// proves the CUDA runtime is genuinely wired up rather than merely linked.
TEST(DeviceBuffer, CudaRoundTrip) {
  if (!CudaAvailable()) GTEST_SKIP() << "no CUDA device available";

  constexpr size_t kN = 4096;
  auto dev = DeviceBuffer::Allocate(kN, DeviceId::Cuda(0));
  ASSERT_TRUE(dev.ok()) << dev.status();

  std::vector<unsigned char> src(kN), dst(kN, 0);
  for (size_t i = 0; i < kN; ++i) src[i] = static_cast<unsigned char>(i & 0xFF);

  ASSERT_EQ(cudaMemcpy(dev->data(), src.data(), kN, cudaMemcpyHostToDevice),
            cudaSuccess);
  ASSERT_EQ(cudaMemcpy(dst.data(), dev->data(), kN, cudaMemcpyDeviceToHost),
            cudaSuccess);
  EXPECT_EQ(src, dst);
}

TEST(DeviceBuffer, OversizedCudaAllocationIsResourceExhausted) {
  if (!CudaAvailable()) GTEST_SKIP() << "no CUDA device available";

  // Far beyond any current device; must surface as ResourceExhausted so the
  // scheduler's preemption path can distinguish it from an internal fault.
  auto b = DeviceBuffer::Allocate(size_t{1} << 46, DeviceId::Cuda(0));
  ASSERT_FALSE(b.ok());
  EXPECT_EQ(b.status().code(), absl::StatusCode::kResourceExhausted);
}
#endif  // INFERX_WITH_CUDA

TEST(CudaUtils, DeviceCountIsSane) {
  const int n = CudaDeviceCount();
  EXPECT_GE(n, 0);
  if (!kCudaEnabled) {
    EXPECT_EQ(n, 0);
    EXPECT_FALSE(CudaAvailable());
  }
}

}  // namespace
}  // namespace inferx
