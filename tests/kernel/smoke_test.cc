#include "inferx/kernels/smoke.h"

#include <vector>

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include "inferx/core/cuda_utils.h"
#include "inferx/core/device.h"
#include "inferx/core/device_buffer.h"

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
