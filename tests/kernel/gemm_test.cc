#include "inferx/kernels/gemm.h"

#include <cmath>
#include <vector>

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include "inferx/core/cuda_utils.h"
#include "inferx/core/device_buffer.h"
#include "inferx/core/shape.h"
#include "inferx/core/tensor_view.h"

namespace inferx {
namespace {

// Deterministic, and deliberately not symmetric in i and j: a mapping bug that
// transposes an operand survives any fill where f(i, j) == f(j, i).
float Fill(int64_t i, int64_t j, float salt) {
  return std::sin(static_cast<float>(i) * 0.5f + salt) *
         std::cos(static_cast<float>(j) * 0.25f + salt);
}

std::vector<__half> ToHalf(const std::vector<float>& v) {
  std::vector<__half> h(v.size());
  for (size_t i = 0; i < v.size(); ++i) h[i] = __float2half(v[i]);
  return h;
}

// y[m, n] = x[m, k] · w[n, k]ᵀ, in f32 on the host.
std::vector<float> ReferenceLinear(const std::vector<float>& x,
                                   const std::vector<float>& w, int64_t m,
                                   int64_t n, int64_t k) {
  std::vector<float> y(static_cast<size_t>(m * n), 0.0f);

  for (int64_t i = 0; i < m; ++i) {
    for (int64_t j = 0; j < n; ++j) {
      float acc = 0.0f;
      for (int64_t p = 0; p < k; ++p) acc += x[i * k + p] * w[j * k + p];
      y[i * n + j] = acc;
    }
  }

  return y;
}

class GemmTest : public ::testing::Test {
 protected:
  void SetUp() override {
    if (!CudaAvailable()) GTEST_SKIP() << "no CUDA device available";
  }

  // Runs one shape end to end and compares against the host reference.
  void CheckShape(int64_t m, int64_t n, int64_t k) {
    auto gemm = kernels::CublasLtGemm::Create();
    ASSERT_TRUE(gemm.ok()) << gemm.status();

    std::vector<float> xf(static_cast<size_t>(m * k));
    std::vector<float> wf(static_cast<size_t>(n * k));

    for (int64_t i = 0; i < m; ++i)
      for (int64_t p = 0; p < k; ++p) xf[i * k + p] = Fill(i, p, 0.0f);
    for (int64_t j = 0; j < n; ++j)
      for (int64_t p = 0; p < k; ++p) wf[j * k + p] = Fill(j, p, 1.7f);

    const std::vector<__half> xh = ToHalf(xf);
    const std::vector<__half> wh = ToHalf(wf);

    auto xb = DeviceBuffer::Allocate(xh.size() * sizeof(__half),
                                     DeviceId::Cuda(0));
    auto wb = DeviceBuffer::Allocate(wh.size() * sizeof(__half),
                                     DeviceId::Cuda(0));
    auto yb = DeviceBuffer::Allocate(static_cast<size_t>(m * n) * sizeof(__half),
                                     DeviceId::Cuda(0));
    ASSERT_TRUE(xb.ok() && wb.ok() && yb.ok());

    ASSERT_EQ(cudaMemcpy(xb->data(), xh.data(), xh.size() * sizeof(__half),
                         cudaMemcpyHostToDevice),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpy(wb->data(), wh.data(), wh.size() * sizeof(__half),
                         cudaMemcpyHostToDevice),
              cudaSuccess);

    auto x = TensorView::Create(xb->data(), DataType::kFloat16, Shape({m, k}),
                                DeviceId::Cuda(0));
    auto w = TensorView::Create(wb->data(), DataType::kFloat16, Shape({n, k}),
                                DeviceId::Cuda(0));
    auto y = TensorView::Create(yb->data(), DataType::kFloat16, Shape({m, n}),
                                DeviceId::Cuda(0));
    ASSERT_TRUE(x.ok() && w.ok() && y.ok());

    const Status s = gemm->LinearF16(*x, *w, *y);
    ASSERT_TRUE(s.ok()) << s;
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    std::vector<__half> got(static_cast<size_t>(m * n));
    ASSERT_EQ(cudaMemcpy(got.data(), yb->data(), got.size() * sizeof(__half),
                         cudaMemcpyDeviceToHost),
              cudaSuccess);

    // Inputs are rounded to f16 before the GEMM and the result is rounded again
    // on the way out, so the tolerance tracks the f16 mantissa (2^-11) scaled by
    // how many terms the dot product accumulated.
    const float tol = 2e-2f + 4e-3f * std::sqrt(static_cast<float>(k));
    const std::vector<float> want = ReferenceLinear(xf, wf, m, n, k);

    for (int64_t i = 0; i < m; ++i) {
      for (int64_t j = 0; j < n; ++j) {
        const float g = __half2float(got[i * n + j]);
        ASSERT_NEAR(g, want[i * n + j], tol)
            << "at (" << i << ", " << j << ") shape m=" << m << " n=" << n
            << " k=" << k;
      }
    }
  }
};

// Square, and small enough that a transposition bug cannot hide behind a
// coincidentally-compatible shape.
TEST_F(GemmTest, MatchesReferenceSquare) { CheckShape(32, 32, 32); }

// All three extents distinct, so any operand swap in the column-major mapping
// is a shape error rather than a wrong-but-plausible result.
TEST_F(GemmTest, MatchesReferenceRectangular) { CheckShape(17, 40, 23); }

// The decode case: one token through a 7B-ish projection. This is the shape the
// engine spends most of its life in, and it is memory-bound rather than
// compute-bound, which is why it earns its own test.
TEST_F(GemmTest, MatchesReferenceDecodeShape) { CheckShape(1, 512, 3584); }

TEST_F(GemmTest, PlansAreCachedPerShape) {
  auto gemm = kernels::CublasLtGemm::Create();
  ASSERT_TRUE(gemm.ok()) << gemm.status();
  EXPECT_EQ(gemm->PlanCacheSize(), 0u);

  ASSERT_TRUE(gemm->Warm(16, 32, 64).ok());
  EXPECT_EQ(gemm->PlanCacheSize(), 1u);

  // Same shape: served from the cache, not planned again.
  ASSERT_TRUE(gemm->Warm(16, 32, 64).ok());
  EXPECT_EQ(gemm->PlanCacheSize(), 1u);

  ASSERT_TRUE(gemm->Warm(16, 32, 128).ok());
  EXPECT_EQ(gemm->PlanCacheSize(), 2u);
}

TEST_F(GemmTest, MismatchedInnerDimIsRejected) {
  auto gemm = kernels::CublasLtGemm::Create();
  ASSERT_TRUE(gemm.ok()) << gemm.status();

  auto buf = DeviceBuffer::Allocate(64 * 64 * sizeof(__half), DeviceId::Cuda(0));
  ASSERT_TRUE(buf.ok());

  auto x = TensorView::Create(buf->data(), DataType::kFloat16, Shape({8, 16}),
                              DeviceId::Cuda(0));
  auto w = TensorView::Create(buf->data(), DataType::kFloat16, Shape({4, 32}),
                              DeviceId::Cuda(0));  // k=32 != 16
  auto y = TensorView::Create(buf->data(), DataType::kFloat16, Shape({8, 4}),
                              DeviceId::Cuda(0));
  ASSERT_TRUE(x.ok() && w.ok() && y.ok());

  EXPECT_EQ(gemm->LinearF16(*x, *w, *y).code(),
            absl::StatusCode::kInvalidArgument);
}

TEST_F(GemmTest, WrongDtypeIsRejected) {
  auto gemm = kernels::CublasLtGemm::Create();
  ASSERT_TRUE(gemm.ok()) << gemm.status();

  auto buf = DeviceBuffer::Allocate(64 * 64 * sizeof(float), DeviceId::Cuda(0));
  ASSERT_TRUE(buf.ok());

  auto f32 = TensorView::Create(buf->data(), DataType::kFloat, Shape({8, 16}),
                                DeviceId::Cuda(0));
  auto f16 = TensorView::Create(buf->data(), DataType::kFloat16, Shape({8, 16}),
                                DeviceId::Cuda(0));
  ASSERT_TRUE(f32.ok() && f16.ok());

  EXPECT_EQ(gemm->LinearF16(*f32, *f16, *f16).code(),
            absl::StatusCode::kInvalidArgument);
}

}  // namespace
}  // namespace inferx
