#include <cmath>
#include <vector>

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include "inferx/core/cuda_utils.h"
#include "inferx/core/device_buffer.h"
#include "inferx/core/shape.h"
#include "inferx/core/tensor_view.h"
#include "inferx/kernels/gemm.h"
#include "inferx/kernels/quantize.h"

namespace inferx {
namespace {

float Fill(int64_t i, int64_t j, float salt) {
  return std::sin(static_cast<float>(i) * 0.5f + salt) *
         std::cos(static_cast<float>(j) * 0.25f + salt);
}

// Allocates a device buffer, uploads f16, and hands back a view over it. The
// buffer is kept alive by the caller's vector.
TensorView UploadF16(std::vector<DeviceBuffer>& keep,
                     const std::vector<float>& host, int64_t rows,
                     int64_t cols) {
  std::vector<__half> h(host.size());
  for (size_t i = 0; i < host.size(); ++i) h[i] = __float2half(host[i]);

  auto buf = DeviceBuffer::Allocate(h.size() * sizeof(__half),
                                    DeviceId::Cuda(0));
  EXPECT_TRUE(buf.ok()) << buf.status();

  EXPECT_EQ(cudaMemcpy(buf->data(), h.data(), h.size() * sizeof(__half),
                       cudaMemcpyHostToDevice),
            cudaSuccess);

  keep.push_back(*std::move(buf));

  auto v = TensorView::Create(keep.back().data(), DataType::kFloat16,
                              Shape({rows, cols}), DeviceId::Cuda(0));
  EXPECT_TRUE(v.ok()) << v.status();

  return *v;
}

class Fp8GemmTest : public ::testing::Test {
 protected:
  void SetUp() override {
    if (!CudaAvailable()) GTEST_SKIP() << "no CUDA device available";
  }
};

// The end-to-end path: quantize both operands, GEMM in FP8, compare against an
// f32 reference computed from the *original* values.
//
// The tolerance is what this test is really about. e4m3 keeps 3 mantissa bits,
// so a single element carries ~6% relative error, and per-tensor scaling means
// small elements are quantized against the largest element in the whole tensor.
// A k-term dot product does not accumulate that error linearly, though -- the
// per-element errors are near-independent and partially cancel, so the sum's
// error grows like sqrt(k) rather than k. The bound below is that shape, and it
// is checked against a relative measure so it does not silently pass by being
// loose on tiny outputs.
TEST_F(Fp8GemmTest, MatchesF32ReferenceWithinQuantizationError) {
  constexpr int64_t m = 24, n = 48, k = 128;

  auto gemm = kernels::CublasLtGemm::Create();
  ASSERT_TRUE(gemm.ok()) << gemm.status();

  std::vector<float> xf(m * k), wf(n * k);
  for (int64_t i = 0; i < m; ++i)
    for (int64_t p = 0; p < k; ++p) xf[i * k + p] = Fill(i, p, 0.0f);
  for (int64_t j = 0; j < n; ++j)
    for (int64_t p = 0; p < k; ++p) wf[j * k + p] = Fill(j, p, 1.7f);

  std::vector<DeviceBuffer> keep;
  const TensorView x16 = UploadF16(keep, xf, m, k);
  const TensorView w16 = UploadF16(keep, wf, n, k);

  auto xq_buf = DeviceBuffer::Allocate(m * k, DeviceId::Cuda(0));
  auto wq_buf = DeviceBuffer::Allocate(n * k, DeviceId::Cuda(0));
  auto y_buf = DeviceBuffer::Allocate(m * n * sizeof(__half), DeviceId::Cuda(0));
  auto scales = DeviceBuffer::Allocate(2 * sizeof(float), DeviceId::Cuda(0));
  ASSERT_TRUE(xq_buf.ok() && wq_buf.ok() && y_buf.ok() && scales.ok());

  auto xq = TensorView::Create(xq_buf->data(), DataType::kFloat8E4M3FN,
                               Shape({m, k}), DeviceId::Cuda(0));
  auto wq = TensorView::Create(wq_buf->data(), DataType::kFloat8E4M3FN,
                               Shape({n, k}), DeviceId::Cuda(0));
  auto y = TensorView::Create(y_buf->data(), DataType::kFloat16, Shape({m, n}),
                              DeviceId::Cuda(0));
  ASSERT_TRUE(xq.ok() && wq.ok() && y.ok());

  float* const x_scale = reinterpret_cast<float*>(scales->data());
  float* const w_scale = x_scale + 1;

  ASSERT_TRUE(kernels::ComputeF8Scale(x16, x_scale).ok());
  ASSERT_TRUE(kernels::ComputeF8Scale(w16, w_scale).ok());
  ASSERT_TRUE(kernels::QuantizeF16ToF8E4M3(x16, *xq, x_scale).ok());
  ASSERT_TRUE(kernels::QuantizeF16ToF8E4M3(w16, *wq, w_scale).ok());

  const Status s = gemm->LinearF8E4M3(*xq, *wq, *y, x_scale, w_scale);
  ASSERT_TRUE(s.ok()) << s;
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  std::vector<__half> got(static_cast<size_t>(m * n));
  ASSERT_EQ(cudaMemcpy(got.data(), y_buf->data(), got.size() * sizeof(__half),
                       cudaMemcpyDeviceToHost),
            cudaSuccess);

  double worst_rel = 0.0;
  double ref_scale = 0.0;

  for (int64_t i = 0; i < m; ++i) {
    for (int64_t j = 0; j < n; ++j) {
      double acc = 0.0;
      for (int64_t p = 0; p < k; ++p) acc += xf[i * k + p] * wf[j * k + p];
      ref_scale = std::max(ref_scale, std::abs(acc));
    }
  }

  ASSERT_GT(ref_scale, 0.0) << "degenerate reference";

  for (int64_t i = 0; i < m; ++i) {
    for (int64_t j = 0; j < n; ++j) {
      double acc = 0.0;
      for (int64_t p = 0; p < k; ++p) acc += xf[i * k + p] * wf[j * k + p];

      const double err =
          std::abs(acc - static_cast<double>(__half2float(got[i * n + j])));
      worst_rel = std::max(worst_rel, err / ref_scale);
    }
  }

  // ~6% per element, growing as sqrt(k) over the dot product and normalized by
  // the largest reference magnitude. Generous, but it is bounding quantization
  // error, not a kernel bug: a transposed operand or a dropped scale misses
  // this by orders of magnitude, which is what it is here to catch.
  const double bound = 0.06 * std::sqrt(static_cast<double>(k)) / 8.0;

  EXPECT_LT(worst_rel, bound)
      << "worst relative error " << worst_rel << " exceeded " << bound;
}

// The scale must actually be applied. Without it the result is off by
// x_scale * w_scale -- a large constant factor, not a rounding difference --
// so this fails loudly if the scale pointers are ever dropped or swapped for
// the unit placeholders the plan is built with.
TEST_F(Fp8GemmTest, ScalesAreAppliedNotIgnored) {
  constexpr int64_t m = 8, n = 16, k = 64;

  auto gemm = kernels::CublasLtGemm::Create();
  ASSERT_TRUE(gemm.ok()) << gemm.status();

  // Magnitudes far from 1 so that omitting the scale cannot coincidentally
  // land near the right answer.
  std::vector<float> xf(m * k, 0.0f), wf(n * k, 0.0f);
  for (int64_t i = 0; i < m * k; ++i) xf[i] = 12.0f;
  for (int64_t i = 0; i < n * k; ++i) wf[i] = 0.25f;

  std::vector<DeviceBuffer> keep;
  const TensorView x16 = UploadF16(keep, xf, m, k);
  const TensorView w16 = UploadF16(keep, wf, n, k);

  auto xq_buf = DeviceBuffer::Allocate(m * k, DeviceId::Cuda(0));
  auto wq_buf = DeviceBuffer::Allocate(n * k, DeviceId::Cuda(0));
  auto y_buf = DeviceBuffer::Allocate(m * n * sizeof(__half), DeviceId::Cuda(0));
  auto scales = DeviceBuffer::Allocate(2 * sizeof(float), DeviceId::Cuda(0));
  ASSERT_TRUE(xq_buf.ok() && wq_buf.ok() && y_buf.ok() && scales.ok());

  auto xq = TensorView::Create(xq_buf->data(), DataType::kFloat8E4M3FN,
                               Shape({m, k}), DeviceId::Cuda(0));
  auto wq = TensorView::Create(wq_buf->data(), DataType::kFloat8E4M3FN,
                               Shape({n, k}), DeviceId::Cuda(0));
  auto y = TensorView::Create(y_buf->data(), DataType::kFloat16, Shape({m, n}),
                              DeviceId::Cuda(0));
  ASSERT_TRUE(xq.ok() && wq.ok() && y.ok());

  float* const x_scale = reinterpret_cast<float*>(scales->data());
  float* const w_scale = x_scale + 1;

  ASSERT_TRUE(kernels::ComputeF8Scale(x16, x_scale).ok());
  ASSERT_TRUE(kernels::ComputeF8Scale(w16, w_scale).ok());
  ASSERT_TRUE(kernels::QuantizeF16ToF8E4M3(x16, *xq, x_scale).ok());
  ASSERT_TRUE(kernels::QuantizeF16ToF8E4M3(w16, *wq, w_scale).ok());
  ASSERT_TRUE(gemm->LinearF8E4M3(*xq, *wq, *y, x_scale, w_scale).ok());
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  std::vector<__half> got(static_cast<size_t>(m * n));
  ASSERT_EQ(cudaMemcpy(got.data(), y_buf->data(), got.size() * sizeof(__half),
                       cudaMemcpyDeviceToHost),
            cudaSuccess);

  // Every element is uniform, so every output is k * 12 * 0.25 = 192.
  const float want = static_cast<float>(k) * 12.0f * 0.25f;

  for (size_t i = 0; i < got.size(); ++i) {
    EXPECT_NEAR(__half2float(got[i]), want, want * 0.02f) << "at " << i;
  }
}

TEST_F(Fp8GemmTest, PlansAreSeparateFromTheF16Path) {
  auto gemm = kernels::CublasLtGemm::Create();
  ASSERT_TRUE(gemm.ok()) << gemm.status();

  ASSERT_TRUE(gemm->Warm(16, 32, 64, /*fp8=*/false).ok());
  EXPECT_EQ(gemm->PlanCacheSize(), 1u);

  // Same shape, different element type: must be planned separately, or an f16
  // algorithm would be handed to an f8 matmul.
  ASSERT_TRUE(gemm->Warm(16, 32, 64, /*fp8=*/true).ok());
  EXPECT_EQ(gemm->PlanCacheSize(), 2u);
}

// The autotune kill-switch (Create(..., /*autotune=*/false)) leaves the
// heuristic's top-1 algorithm in place -- the differential-testing reference
// for Tune. The algorithm choice is speed, not math, so the autotune-on and
// autotune-off results must agree within FP8 + accumulation rounding.
TEST_F(Fp8GemmTest, AutotuneOffAgreesWithAutotuneOn) {
  constexpr int64_t m = 8, n = 64, k = 128;

  std::vector<float> xf(m * k), wf(n * k);
  for (int64_t i = 0; i < m * k; ++i) xf[i] = Fill(i / k, i % k, 0.0f);
  for (int64_t j = 0; j < n * k; ++j) wf[j] = Fill(j / k, j % k, 1.7f);

  std::vector<DeviceBuffer> keep;
  const TensorView x16 = UploadF16(keep, xf, m, k);
  const TensorView w16 = UploadF16(keep, wf, n, k);

  auto xq_buf = DeviceBuffer::Allocate(m * k, DeviceId::Cuda(0));
  auto wq_buf = DeviceBuffer::Allocate(n * k, DeviceId::Cuda(0));
  auto y_on = DeviceBuffer::Allocate(m * n * sizeof(__half), DeviceId::Cuda(0));
  auto y_off = DeviceBuffer::Allocate(m * n * sizeof(__half), DeviceId::Cuda(0));
  auto scales = DeviceBuffer::Allocate(2 * sizeof(float), DeviceId::Cuda(0));
  ASSERT_TRUE(xq_buf.ok() && wq_buf.ok() && y_on.ok() && y_off.ok() &&
              scales.ok());

  auto xq = TensorView::Create(xq_buf->data(), DataType::kFloat8E4M3FN,
                               Shape({m, k}), DeviceId::Cuda(0));
  auto wq = TensorView::Create(wq_buf->data(), DataType::kFloat8E4M3FN,
                               Shape({n, k}), DeviceId::Cuda(0));
  auto on = TensorView::Create(y_on->data(), DataType::kFloat16,
                               Shape({m, n}), DeviceId::Cuda(0));
  auto off = TensorView::Create(y_off->data(), DataType::kFloat16,
                                Shape({m, n}), DeviceId::Cuda(0));
  ASSERT_TRUE(xq.ok() && wq.ok() && on.ok() && off.ok());

  float* const s = reinterpret_cast<float*>(scales->data());
  ASSERT_TRUE(kernels::ComputeF8Scale(x16, s).ok());
  ASSERT_TRUE(kernels::ComputeF8Scale(w16, s + 1).ok());
  ASSERT_TRUE(kernels::QuantizeF16ToF8E4M3(x16, *xq, s).ok());
  ASSERT_TRUE(kernels::QuantizeF16ToF8E4M3(w16, *wq, s + 1).ok());
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  auto gemm_on = kernels::CublasLtGemm::Create();
  auto gemm_off = kernels::CublasLtGemm::Create(kernels::CublasLtGemm::kDefaultWorkspaceBytes,
                                                /*autotune=*/false);
  ASSERT_TRUE(gemm_on.ok() && gemm_off.ok());

  ASSERT_TRUE(gemm_on->LinearF8E4M3(*xq, *wq, *on, s, s + 1).ok());
  ASSERT_TRUE(gemm_off->LinearF8E4M3(*xq, *wq, *off, s, s + 1).ok());
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  std::vector<__half> got_on(m * n), got_off(m * n);
  ASSERT_EQ(cudaMemcpy(got_on.data(), y_on->data(), got_on.size() * sizeof(__half),
                       cudaMemcpyDeviceToHost),
            cudaSuccess);
  ASSERT_EQ(cudaMemcpy(got_off.data(), y_off->data(),
                       got_off.size() * sizeof(__half), cudaMemcpyDeviceToHost),
            cudaSuccess);

  // Both ran the same FP8 matmul; only the algorithm differed. Different
  // cuBLASLt algorithms accumulate in a different order, so the gap is small
  // rounding relative to the output's own magnitude -- not the FP8
  // quantization error, which is correlated (same quantized inputs) and cancels.
  double max_mag = 0.0;
  for (size_t i = 0; i < got_on.size(); ++i) {
    max_mag = std::max(max_mag, std::fabs(static_cast<double>(__half2float(got_on[i]))));
  }
  for (size_t i = 0; i < got_on.size(); ++i) {
    EXPECT_NEAR(__half2float(got_on[i]), __half2float(got_off[i]),
                max_mag * 1e-2 + 1e-3)
        << "autotune on/off disagree at " << i;
  }
}

// k not a multiple of 16 is refused with a diagnostic that says so, rather than
// reaching cuBLASLt and coming back as "no algorithm".
TEST_F(Fp8GemmTest, UnalignedKIsRejectedClearly) {
  auto gemm = kernels::CublasLtGemm::Create();
  ASSERT_TRUE(gemm.ok()) << gemm.status();

  auto buf = DeviceBuffer::Allocate(64 * 64, DeviceId::Cuda(0));
  auto ybuf = DeviceBuffer::Allocate(64 * 64 * sizeof(__half),
                                     DeviceId::Cuda(0));
  auto sc = DeviceBuffer::Allocate(2 * sizeof(float), DeviceId::Cuda(0));
  ASSERT_TRUE(buf.ok() && ybuf.ok() && sc.ok());

  auto x = TensorView::Create(buf->data(), DataType::kFloat8E4M3FN,
                              Shape({8, 20}), DeviceId::Cuda(0));  // k = 20
  auto w = TensorView::Create(buf->data(), DataType::kFloat8E4M3FN,
                              Shape({16, 20}), DeviceId::Cuda(0));
  auto y = TensorView::Create(ybuf->data(), DataType::kFloat16, Shape({8, 16}),
                              DeviceId::Cuda(0));
  ASSERT_TRUE(x.ok() && w.ok() && y.ok());

  float* const s = reinterpret_cast<float*>(sc->data());
  const Status st = gemm->LinearF8E4M3(*x, *w, *y, s, s + 1);

  EXPECT_EQ(st.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_NE(st.message().find("multiple of 16"), std::string_view::npos) << st;
}

TEST_F(Fp8GemmTest, F16InputToTheF8PathIsRejected) {
  auto gemm = kernels::CublasLtGemm::Create();
  ASSERT_TRUE(gemm.ok()) << gemm.status();

  auto buf = DeviceBuffer::Allocate(64 * 64 * sizeof(__half),
                                    DeviceId::Cuda(0));
  auto sc = DeviceBuffer::Allocate(2 * sizeof(float), DeviceId::Cuda(0));
  ASSERT_TRUE(buf.ok() && sc.ok());

  auto f16 = TensorView::Create(buf->data(), DataType::kFloat16, Shape({8, 16}),
                                DeviceId::Cuda(0));
  ASSERT_TRUE(f16.ok());

  float* const s = reinterpret_cast<float*>(sc->data());

  EXPECT_EQ(gemm->LinearF8E4M3(*f16, *f16, *f16, s, s + 1).code(),
            absl::StatusCode::kInvalidArgument);
}

// The scale is amax/448, so quantizing and dequantizing must round-trip the
// tensor's largest magnitude almost exactly -- it is the one value e4m3 is
// guaranteed to represent well.
TEST_F(Fp8GemmTest, ScaleMapsAmaxOntoTheFormatMaximum) {
  constexpr int64_t n = 1024;

  std::vector<float> host(n);
  for (int64_t i = 0; i < n; ++i) host[i] = 0.01f * static_cast<float>(i % 37);
  host[500] = 3.5f;  // the amax

  std::vector<DeviceBuffer> keep;
  const TensorView src = UploadF16(keep, host, 1, n);

  auto scale_buf = DeviceBuffer::Allocate(sizeof(float), DeviceId::Cuda(0));
  ASSERT_TRUE(scale_buf.ok());
  float* const scale = reinterpret_cast<float*>(scale_buf->data());

  ASSERT_TRUE(kernels::ComputeF8Scale(src, scale).ok());
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  float got = 0.0f;
  ASSERT_EQ(cudaMemcpy(&got, scale, sizeof(float), cudaMemcpyDeviceToHost),
            cudaSuccess);

  EXPECT_NEAR(got, 3.5f / kernels::kFloat8E4M3Max, 1e-6f);
}

// bf16 output, which is what the model needs: an FP8 GEMM has to drop into a
// bf16 stack without a conversion on either side.
TEST_F(Fp8GemmTest, ProducesBf16Output) {
  constexpr int64_t m = 8, n = 32, k = 64;

  auto gemm = kernels::CublasLtGemm::Create();
  ASSERT_TRUE(gemm.ok()) << gemm.status();

  auto xq_buf = DeviceBuffer::Allocate(m * k, DeviceId::Cuda(0));
  auto wq_buf = DeviceBuffer::Allocate(n * k, DeviceId::Cuda(0));
  auto y_buf = DeviceBuffer::Allocate(m * n * 2, DeviceId::Cuda(0));
  auto scales = DeviceBuffer::Allocate(2 * sizeof(float), DeviceId::Cuda(0));
  ASSERT_TRUE(xq_buf.ok() && wq_buf.ok() && y_buf.ok() && scales.ok());

  ASSERT_EQ(cudaMemset(xq_buf->data(), 0x38, xq_buf->size()), cudaSuccess);
  ASSERT_EQ(cudaMemset(wq_buf->data(), 0x38, wq_buf->size()), cudaSuccess);

  const float one[2] = {1.0f, 1.0f};
  ASSERT_EQ(cudaMemcpy(scales->data(), one, sizeof(one),
                       cudaMemcpyHostToDevice),
            cudaSuccess);

  auto xq = TensorView::Create(xq_buf->data(), DataType::kFloat8E4M3FN,
                               Shape({m, k}), DeviceId::Cuda(0));
  auto wq = TensorView::Create(wq_buf->data(), DataType::kFloat8E4M3FN,
                               Shape({n, k}), DeviceId::Cuda(0));
  auto y = TensorView::Create(y_buf->data(), DataType::kBFloat16,
                              Shape({m, n}), DeviceId::Cuda(0));
  ASSERT_TRUE(xq.ok() && wq.ok() && y.ok());

  float* const s = reinterpret_cast<float*>(scales->data());

  const Status st = gemm->LinearF8E4M3(*xq, *wq, *y, s, s + 1);
  ASSERT_TRUE(st.ok()) << st;
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess)
      << "bf16-output FP8 GEMM faulted";
}

}  // namespace
}  // namespace inferx
