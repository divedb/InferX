#include "inferx/kernels/smoke.h"

#include <vector>

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include "inferx/core/cuda_utils.h"
#include "inferx/core/device.h"
#include "inferx/core/device_buffer.h"
#include "inferx/core/shape.h"
#include "inferx/core/tensor_view.h"

namespace inferx {
namespace {

// The test that answers "is the device path real". It is a round trip on
// purpose: a kernel that compiles and launches but computes nothing would still
// pass a launch-status-only check.
TEST(KernelSmoke, ScaleRoundTrip) {
  if (!CudaAvailable()) GTEST_SKIP() << "no CUDA device available";

  constexpr int64_t kN = 1000;  // not a multiple of the block size, on purpose

  std::vector<float> host(kN);
  for (int64_t i = 0; i < kN; ++i) host[i] = static_cast<float>(i);

  auto buf = DeviceBuffer::Allocate(kN * sizeof(float), DeviceId::Cuda(0));
  ASSERT_TRUE(buf.ok()) << buf.status();

  ASSERT_EQ(cudaMemcpy(buf->data(), host.data(), kN * sizeof(float),
                       cudaMemcpyHostToDevice),
            cudaSuccess);

  const Status s =
      kernels::LaunchScale(reinterpret_cast<float*>(buf->data()), kN, 2.0f);
  ASSERT_TRUE(s.ok()) << s;

  std::vector<float> result(kN, -1.0f);
  ASSERT_EQ(cudaMemcpy(result.data(), buf->data(), kN * sizeof(float),
                       cudaMemcpyDeviceToHost),
            cudaSuccess);

  for (int64_t i = 0; i < kN; ++i) {
    ASSERT_FLOAT_EQ(result[i], static_cast<float>(i) * 2.0f) << "at " << i;
  }
}

// The tail of the grid must not write past the end. A kernel missing its bounds
// check passes ScaleRoundTrip and corrupts whatever follows it in the arena,
// which is the failure mode worth catching before real kernels arrive.
TEST(KernelSmoke, DoesNotWritePastTheEnd) {
  if (!CudaAvailable()) GTEST_SKIP() << "no CUDA device available";

  constexpr int64_t kN = 100;
  constexpr int64_t kGuard = 64;

  std::vector<float> host(kN + kGuard, 1.0f);

  auto buf =
      DeviceBuffer::Allocate((kN + kGuard) * sizeof(float), DeviceId::Cuda(0));
  ASSERT_TRUE(buf.ok()) << buf.status();

  ASSERT_EQ(cudaMemcpy(buf->data(), host.data(), (kN + kGuard) * sizeof(float),
                       cudaMemcpyHostToDevice),
            cudaSuccess);

  // Scales only the first kN elements; the guard region must be untouched.
  const Status s =
      kernels::LaunchScale(reinterpret_cast<float*>(buf->data()), kN, 3.0f);
  ASSERT_TRUE(s.ok()) << s;

  std::vector<float> result(kN + kGuard, -1.0f);
  ASSERT_EQ(cudaMemcpy(result.data(), buf->data(),
                       (kN + kGuard) * sizeof(float), cudaMemcpyDeviceToHost),
            cudaSuccess);

  for (int64_t i = 0; i < kN; ++i)
    ASSERT_FLOAT_EQ(result[i], 3.0f) << "at " << i;
  for (int64_t i = kN; i < kN + kGuard; ++i) {
    ASSERT_FLOAT_EQ(result[i], 1.0f) << "guard clobbered at " << i;
  }
}

// The kernel-boundary claim, exercised rather than asserted: the view crosses
// the launch by value, and the kernel finds its extent and data through the
// annotated accessors. A rank-2 shape on purpose -- Numel() has to fold the
// extents device-side, which a rank-1 view would not catch.
TEST(KernelSmoke, TensorViewCrossesTheLaunchBoundary) {
  if (!CudaAvailable()) GTEST_SKIP() << "no CUDA device available";

  constexpr int64_t kRows = 7;
  constexpr int64_t kCols = 19;  // 133 elements: not a multiple of the block
  constexpr int64_t kN = kRows * kCols;

  std::vector<float> host(kN);
  for (int64_t i = 0; i < kN; ++i) host[i] = static_cast<float>(i);

  auto buf = DeviceBuffer::Allocate(kN * sizeof(float), DeviceId::Cuda(0));
  ASSERT_TRUE(buf.ok()) << buf.status();

  ASSERT_EQ(cudaMemcpy(buf->data(), host.data(), kN * sizeof(float),
                       cudaMemcpyHostToDevice),
            cudaSuccess);

  auto view = TensorView::Create(buf->data(), DataType::kFloat,
                                 Shape({kRows, kCols}), DeviceId::Cuda(0));
  ASSERT_TRUE(view.ok()) << view.status();

  const Status s = kernels::LaunchScale(*view, 4.0f);
  ASSERT_TRUE(s.ok()) << s;

  std::vector<float> result(kN, -1.0f);
  ASSERT_EQ(cudaMemcpy(result.data(), buf->data(), kN * sizeof(float),
                       cudaMemcpyDeviceToHost),
            cudaSuccess);

  for (int64_t i = 0; i < kN; ++i) {
    ASSERT_FLOAT_EQ(result[i], static_cast<float>(i) * 4.0f) << "at " << i;
  }
}

// A host-side view must be rejected, not silently dereferenced on the device.
TEST(KernelSmoke, HostViewIsRejected) {
  std::vector<float> host(16, 1.0f);

  auto view = TensorView::Create(host.data(), DataType::kFloat, Shape({16}),
                                 DeviceId::Cpu());
  ASSERT_TRUE(view.ok()) << view.status();

  const Status s = kernels::LaunchScale(*view, 2.0f);
  EXPECT_EQ(s.code(), absl::StatusCode::kInvalidArgument) << s;
}

// Likewise a dtype the kernel cannot read. Checked on the host, where the dtype
// is known, rather than per-thread on the device.
TEST(KernelSmoke, WrongDtypeIsRejected) {
  if (!CudaAvailable()) GTEST_SKIP() << "no CUDA device available";

  auto buf = DeviceBuffer::Allocate(16 * sizeof(int32_t), DeviceId::Cuda(0));
  ASSERT_TRUE(buf.ok()) << buf.status();

  auto view = TensorView::Create(buf->data(), DataType::kInt32, Shape({16}),
                                 DeviceId::Cuda(0));
  ASSERT_TRUE(view.ok()) << view.status();

  const Status s = kernels::LaunchScale(*view, 2.0f);
  EXPECT_EQ(s.code(), absl::StatusCode::kInvalidArgument) << s;
}

TEST(KernelSmoke, EmptyLaunchIsOk) {
  if (!CudaAvailable()) GTEST_SKIP() << "no CUDA device available";

  EXPECT_TRUE(kernels::LaunchScale(nullptr, 0, 2.0f).ok());
}

// Errors come back as a Status, not as a stale sticky error picked up by some
// later unrelated CUDA call.
TEST(KernelSmoke, NullWithWorkIsInvalidArgument) {
  const Status s = kernels::LaunchScale(nullptr, 16, 2.0f);
  EXPECT_EQ(s.code(), absl::StatusCode::kInvalidArgument) << s;
}

}  // namespace
}  // namespace inferx
