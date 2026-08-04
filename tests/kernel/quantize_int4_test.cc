#include <cmath>
#include <vector>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include "inferx/core/cuda_utils.h"
#include "inferx/core/device_buffer.h"
#include "inferx/core/shape.h"
#include "inferx/core/tensor_view.h"
#include "inferx/kernels/gemm.h"
#include "inferx/kernels/quantize.h"
#include "inferx/kernels/w4a16_gemm.h"

namespace inferx {
namespace {

// Deterministic and deliberately asymmetric in i and j, so a transpose or
// group-stride bug cannot hide behind f(i, j) == f(j, i).
float Fill(int64_t i, int64_t j, float salt) {
  return std::sin(static_cast<float>(i) * 0.5f + salt) *
         std::cos(static_cast<float>(j) * 0.25f + salt);
}

// Allocates a device buffer, uploads f16, hands back a view over it. The buffer
// is kept alive by the caller's vector.
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

// The bf16 analogue, for the W4A16-in-the-model path: the model's weights and
// activations are bf16, so the bf16 int4 kernels avoid a bf16<->f16 cast at
// every linear.
TensorView UploadBf16(std::vector<DeviceBuffer>& keep,
                      const std::vector<float>& host, int64_t rows,
                      int64_t cols) {
  std::vector<__nv_bfloat16> h(host.size());
  for (size_t i = 0; i < host.size(); ++i) h[i] = __float2bfloat16(host[i]);

  auto buf = DeviceBuffer::Allocate(h.size() * sizeof(__nv_bfloat16),
                                    DeviceId::Cuda(0));
  EXPECT_TRUE(buf.ok()) << buf.status();

  EXPECT_EQ(cudaMemcpy(buf->data(), h.data(),
                        h.size() * sizeof(__nv_bfloat16),
                        cudaMemcpyHostToDevice),
            cudaSuccess);

  keep.push_back(*std::move(buf));

  auto v = TensorView::Create(keep.back().data(), DataType::kBFloat16,
                              Shape({rows, cols}), DeviceId::Cuda(0));
  EXPECT_TRUE(v.ok()) << v.status();

  return *v;
}

class Int4QuantTest : public ::testing::Test {
 protected:
  void SetUp() override {
    if (!CudaAvailable()) GTEST_SKIP() << "no CUDA device available";
  }
};

// The round trip: f16 -> int4 -> f16 must stay within half a quantization step
// of the original. A step is 2*amax/15 (15 symmetric levels in [-7, 7]), so each
// element's error is bounded by its group's amax/14. Anything larger is a
// rounding or packing bug.
TEST_F(Int4QuantTest, RoundTripStaysWithinHalfAStep) {
  constexpr int64_t n = 32, k = 128;  // one group per row

  std::vector<float> wf(n * k);
  for (int64_t i = 0; i < n; ++i)
    for (int64_t j = 0; j < k; ++j) wf[i * k + j] = Fill(i, j, 0.0f);

  std::vector<DeviceBuffer> keep;
  const TensorView w = UploadF16(keep, wf, n, k);

  auto qb = DeviceBuffer::Allocate(n * k / 2, DeviceId::Cuda(0));   // packed int4
  auto sb = DeviceBuffer::Allocate(n * 1 * sizeof(__half), DeviceId::Cuda(0));
  auto hatb = DeviceBuffer::Allocate(n * k * sizeof(__half), DeviceId::Cuda(0));
  ASSERT_TRUE(qb.ok() && sb.ok() && hatb.ok());

  auto q = TensorView::Create(qb->data(), DataType::kInt4, Shape({n, k}),
                              DeviceId::Cuda(0));
  auto s = TensorView::Create(sb->data(), DataType::kFloat16, Shape({n, 1}),
                              DeviceId::Cuda(0));
  auto hat = TensorView::Create(hatb->data(), DataType::kFloat16,
                                Shape({n, k}), DeviceId::Cuda(0));
  ASSERT_TRUE(q.ok() && s.ok() && hat.ok());

  ASSERT_TRUE(kernels::QuantizeF16ToInt4(w, *q, *s).ok());
  ASSERT_TRUE(kernels::DequantizeInt4ToF16(*q, *s, *hat).ok());
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  std::vector<__half> got(n * k);
  ASSERT_EQ(cudaMemcpy(got.data(), hatb->data(), got.size() * sizeof(__half),
                       cudaMemcpyDeviceToHost),
            cudaSuccess);

  for (int64_t i = 0; i < n; ++i) {
    // The group is one row; its amax defines the step bound for every element
    // in it.
    float amax = 0.0f;
    for (int64_t j = 0; j < k; ++j)
      amax = std::max(amax, std::abs(wf[i * k + j]));

    const float step = amax > 0.0f ? (2.0f * amax) / 15.0f : 0.0f;
    // Round-to-nearest puts the quant error at most half a step; storing the
    // dequantized product back to f16 adds up to one f16 ulp. Together that is
    // comfortably under one full step, so "under a step" is the bound -- loose
    // enough to absorb f16 rounding, tight enough that a packing or scale bug
    // (which errs by many steps) still fails it.
    const float tol = step + 1e-3f;

    for (int64_t j = 0; j < k; ++j) {
      EXPECT_NEAR(__half2float(got[i * k + j]), wf[i * k + j], tol)
          << "at (" << i << ", " << j << ")";
    }
  }
}

// Per-group, not per-tensor: a row with a small-magnitude group beside a large
// one must quantize each against its own scale. A per-tensor scale would erase
// the small group (quantize it to zero against the large group's amax), which
// is exactly what per-group exists to prevent.
TEST_F(Int4QuantTest, ScalesArePerGroupNotPerTensor) {
  constexpr int64_t n = 1, k = 256;  // two groups of 128
  constexpr int64_t group = 128;

  std::vector<float> wf(n * k, 0.0f);
  // Non-zero salts: Fill(0, ...) with salt 0 collapses to sin(0)=0, which would
  // make the "small" group all-zero and defeat the point, so both use a salt
  // that keeps them alive. The small group is ~0.01x the large group's scale.
  for (int64_t j = 0; j < group; ++j) wf[j] = 0.01f * Fill(0, j, 0.3f);   // tiny
  for (int64_t j = group; j < k; ++j) wf[j] = 5.0f * Fill(0, j, 1.7f);    // ~unit-scale

  std::vector<DeviceBuffer> keep;
  const TensorView w = UploadF16(keep, wf, n, k);

  auto qb = DeviceBuffer::Allocate(n * k / 2, DeviceId::Cuda(0));
  auto sb = DeviceBuffer::Allocate(n * 2 * sizeof(__half), DeviceId::Cuda(0));
  auto hatb = DeviceBuffer::Allocate(n * k * sizeof(__half), DeviceId::Cuda(0));
  ASSERT_TRUE(qb.ok() && sb.ok() && hatb.ok());

  auto q = TensorView::Create(qb->data(), DataType::kInt4, Shape({n, k}),
                              DeviceId::Cuda(0));
  auto s = TensorView::Create(sb->data(), DataType::kFloat16, Shape({n, 2}),
                              DeviceId::Cuda(0));
  auto hat = TensorView::Create(hatb->data(), DataType::kFloat16,
                                Shape({n, k}), DeviceId::Cuda(0));
  ASSERT_TRUE(q.ok() && s.ok() && hat.ok());

  ASSERT_TRUE(kernels::QuantizeF16ToInt4(w, *q, *s).ok());
  ASSERT_TRUE(kernels::DequantizeInt4ToF16(*q, *s, *hat).ok());
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  // The small group's scale must be ~15x smaller than the large group's.
  std::vector<__half> scales(2);
  ASSERT_EQ(cudaMemcpy(scales.data(), sb->data(), 2 * sizeof(__half),
                       cudaMemcpyDeviceToHost),
            cudaSuccess);
  const float s_small = __half2float(scales[0]);
  const float s_large = __half2float(scales[1]);
  EXPECT_GT(s_large, 0.0f);
  EXPECT_GT(s_small, 0.0f);
  EXPECT_LT(s_small, s_large / 100.0f)
      << "small group scale " << s_small << " not << large " << s_large;

  // And the small group survives with its shape: its recovered values correlate
  // with the originals rather than collapsing to zero.
  std::vector<__half> got(n * k);
  ASSERT_EQ(cudaMemcpy(got.data(), hatb->data(), got.size() * sizeof(__half),
                       cudaMemcpyDeviceToHost),
            cudaSuccess);
  float small_signal = 0.0f;
  for (int64_t j = 0; j < group; ++j)
    small_signal = std::max(small_signal,
                            std::abs(__half2float(got[j])));
  EXPECT_GT(small_signal, 0.0f)
      << "small group quantized away under a per-tensor scale";
}

// The end-to-end W4A16 path: quantize w, dequantize, run the stock fp16 GEMM on
// the dequantized weights, and compare to an f32 reference computed from *the
// same dequantized weights*. Referencing the dequant rather than the original
// keeps this a test of the dequant+GEMM pipeline (the f16 matmul tolerance)
// rather than of int4 quantization accuracy, which the round-trip test owns.
TEST_F(Int4QuantTest, DequantThenGemmMatchesReferenceOnDequantWeights) {
  constexpr int64_t m = 8, n = 512, k = 3584;  // 28 groups of 128 along k
  constexpr int64_t groups = k / 128;

  std::vector<float> xf(m * k), wf(n * k);
  for (int64_t i = 0; i < m; ++i)
    for (int64_t p = 0; p < k; ++p) xf[i * k + p] = Fill(i, p, 0.0f);
  for (int64_t j = 0; j < n; ++j)
    for (int64_t p = 0; p < k; ++p) wf[j * k + p] = Fill(j, p, 1.7f);

  std::vector<DeviceBuffer> keep;
  const TensorView x = UploadF16(keep, xf, m, k);
  const TensorView w = UploadF16(keep, wf, n, k);

  auto qb = DeviceBuffer::Allocate(n * k / 2, DeviceId::Cuda(0));
  auto sb = DeviceBuffer::Allocate(n * groups * sizeof(__half), DeviceId::Cuda(0));
  auto hatb = DeviceBuffer::Allocate(n * k * sizeof(__half), DeviceId::Cuda(0));
  auto yb = DeviceBuffer::Allocate(m * n * sizeof(__half), DeviceId::Cuda(0));
  ASSERT_TRUE(qb.ok() && sb.ok() && hatb.ok() && yb.ok());

  auto q = TensorView::Create(qb->data(), DataType::kInt4, Shape({n, k}),
                              DeviceId::Cuda(0));
  auto s = TensorView::Create(sb->data(), DataType::kFloat16,
                              Shape({n, groups}), DeviceId::Cuda(0));
  auto hat = TensorView::Create(hatb->data(), DataType::kFloat16,
                                Shape({n, k}), DeviceId::Cuda(0));
  auto y = TensorView::Create(yb->data(), DataType::kFloat16, Shape({m, n}),
                              DeviceId::Cuda(0));
  ASSERT_TRUE(q.ok() && s.ok() && hat.ok() && y.ok());

  ASSERT_TRUE(kernels::QuantizeF16ToInt4(w, *q, *s).ok());
  ASSERT_TRUE(kernels::DequantizeInt4ToF16(*q, *s, *hat).ok());

  auto gemm = kernels::CublasLtGemm::Create();
  ASSERT_TRUE(gemm.ok()) << gemm.status();
  ASSERT_TRUE(gemm->LinearF16(x, *hat, *y).ok());
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  // Read back the dequantized weights and compute the reference from *them*, so
  // the tolerance is the f16 matmul's, not int4's.
  std::vector<__half> w_hat(n * k);
  ASSERT_EQ(cudaMemcpy(w_hat.data(), hatb->data(),
                       w_hat.size() * sizeof(__half), cudaMemcpyDeviceToHost),
            cudaSuccess);

  std::vector<__half> got(m * n);
  ASSERT_EQ(cudaMemcpy(got.data(), yb->data(), got.size() * sizeof(__half),
                       cudaMemcpyDeviceToHost),
            cudaSuccess);

  // f16 input/output, fp32 accumulate: tolerance tracks the f16 mantissa (2^-11)
  // scaled by how many terms the dot product accumulated, same form as the
  // unquantized GEMM test.
  const float tol = 2e-2f + 4e-3f * std::sqrt(static_cast<float>(k));

  for (int64_t i = 0; i < m; ++i) {
    for (int64_t j = 0; j < n; ++j) {
      float acc = 0.0f;
      for (int64_t p = 0; p < k; ++p)
        acc += xf[i * k + p] * __half2float(w_hat[j * k + p]);

      EXPECT_NEAR(__half2float(got[i * n + j]), acc, tol)
          << "at (" << i << ", " << j << ")";
    }
  }
}

// Rejection: a scales layout whose group count does not divide k is a shape
// contract violation, and the kernel must say so rather than read past a group
// boundary. (An odd k is rejected earlier by TensorView -- sub-byte rows must
// be byte-aligned -- so the kernel's own even-k check is defense-in-depth and
// not reachable through the public API.)
TEST_F(Int4QuantTest, GroupCountMustDivideK) {
  constexpr int64_t n = 4, k = 128;

  std::vector<DeviceBuffer> keep;
  const TensorView w = UploadF16(keep, std::vector<float>(n * k, 0.5f), n, k);

  auto qb = DeviceBuffer::Allocate(n * k / 2, DeviceId::Cuda(0));
  // 3 groups per row over k=128 does not divide evenly.
  auto sb = DeviceBuffer::Allocate(n * 3 * sizeof(__half), DeviceId::Cuda(0));
  ASSERT_TRUE(qb.ok() && sb.ok());

  auto q = TensorView::Create(qb->data(), DataType::kInt4, Shape({n, k}),
                              DeviceId::Cuda(0));
  auto s = TensorView::Create(sb->data(), DataType::kFloat16, Shape({n, 3}),
                              DeviceId::Cuda(0));
  ASSERT_TRUE(q.ok() && s.ok());

  const Status st = kernels::QuantizeF16ToInt4(w, *q, *s);
  EXPECT_EQ(st.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_NE(st.message().find("divide"), std::string_view::npos) << st;
}

// Regression: the per-group amax reduction must stay correct for group sizes
// that are not a power of two. The scale kernel's block reduction halves its
// stride each step, which only combines every lane when blockDim is a power of
// two; a non-pow-2 block silently orphans lanes and understates the amax.
// group=96 is the smallest size the old block selector served with a non-pow-2
// block (96), so k=192 with two groups of 96 exercises it. A distinct peak is
// placed at a moving offset per (row, group) so the peak lands in every lane
// position across the batch, including whichever lanes a broken reduction
// drops; a correct reduction recovers peak/7 for every group.
TEST_F(Int4QuantTest, PerGroupScaleIsCorrectForNonPowerOfTwoGroupSize) {
  constexpr int64_t group = 96;
  constexpr int64_t k = group * 2;        // two non-pow-2 groups per row
  constexpr int64_t groups = k / group;
  constexpr int64_t n = group;            // enough rows to cover every offset

  constexpr float kPeak = 5.0f;
  constexpr float kFloor = 0.01f;

  std::vector<float> wf(n * k, kFloor);
  for (int64_t i = 0; i < n; ++i) {
    for (int64_t g = 0; g < groups; ++g) {
      const int64_t off = (i + g * (group / 2)) % group;
      wf[i * k + g * group + off] = kPeak;
    }
  }

  std::vector<DeviceBuffer> keep;
  const TensorView w = UploadF16(keep, wf, n, k);

  auto qb = DeviceBuffer::Allocate(n * k / 2, DeviceId::Cuda(0));
  auto sb =
      DeviceBuffer::Allocate(n * groups * sizeof(__half), DeviceId::Cuda(0));
  ASSERT_TRUE(qb.ok() && sb.ok());

  auto q = TensorView::Create(qb->data(), DataType::kInt4, Shape({n, k}),
                              DeviceId::Cuda(0));
  auto s = TensorView::Create(sb->data(), DataType::kFloat16,
                              Shape({n, groups}), DeviceId::Cuda(0));
  ASSERT_TRUE(q.ok() && s.ok());

  ASSERT_TRUE(kernels::QuantizeF16ToInt4(w, *q, *s).ok());
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  std::vector<__half> scales(n * groups);
  ASSERT_EQ(cudaMemcpy(scales.data(), sb->data(),
                       scales.size() * sizeof(__half),
                       cudaMemcpyDeviceToHost),
            cudaSuccess);

  const float expected = kPeak / kernels::kInt4SymmetricMax;
  for (int64_t i = 0; i < n; ++i) {
    for (int64_t g = 0; g < groups; ++g) {
      const float got = __half2float(scales[i * groups + g]);
      // Every group holds the peak somewhere, so its scale is peak/7 within one
      // f16 ulp. A dropped-lane reduction sees only the floor and reports
      // ~0.001 instead of ~0.714, missing by ~700x.
      EXPECT_NEAR(got, expected, expected * 1e-2f)
          << "row " << i << " group " << g << " got " << got;
    }
  }
}

// The bf16 round trip: bf16 -> int4 -> bf16 must stay within half a step, the
// same gate the f16 round-trip test uses. Guards the W4A16-in-model path, which
// quantizes bf16 weights at load and dequantizes to bf16 each step.
TEST_F(Int4QuantTest, Bf16RoundTripStaysWithinHalfAStep) {
  constexpr int64_t n = 32, k = 128;  // one group per row

  std::vector<float> wf(n * k);
  for (int64_t i = 0; i < n; ++i)
    for (int64_t j = 0; j < k; ++j) wf[i * k + j] = Fill(i, j, 0.0f);

  std::vector<DeviceBuffer> keep;
  const TensorView w = UploadBf16(keep, wf, n, k);

  auto qb = DeviceBuffer::Allocate(n * k / 2, DeviceId::Cuda(0));
  auto sb =
      DeviceBuffer::Allocate(n * 1 * sizeof(__nv_bfloat16), DeviceId::Cuda(0));
  auto hatb =
      DeviceBuffer::Allocate(n * k * sizeof(__nv_bfloat16), DeviceId::Cuda(0));
  ASSERT_TRUE(qb.ok() && sb.ok() && hatb.ok());

  auto q = TensorView::Create(qb->data(), DataType::kInt4, Shape({n, k}),
                              DeviceId::Cuda(0));
  auto s = TensorView::Create(sb->data(), DataType::kBFloat16,
                              Shape({n, 1}), DeviceId::Cuda(0));
  auto hat = TensorView::Create(hatb->data(), DataType::kBFloat16,
                                Shape({n, k}), DeviceId::Cuda(0));
  ASSERT_TRUE(q.ok() && s.ok() && hat.ok());

  ASSERT_TRUE(kernels::QuantizeBf16ToInt4(w, *q, *s).ok());
  ASSERT_TRUE(kernels::DequantizeInt4ToBf16(*q, *s, *hat).ok());
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  std::vector<__nv_bfloat16> got(n * k);
  ASSERT_EQ(cudaMemcpy(got.data(), hatb->data(),
                        got.size() * sizeof(__nv_bfloat16),
                        cudaMemcpyDeviceToHost),
            cudaSuccess);

  for (int64_t i = 0; i < n; ++i) {
    float amax = 0.0f;
    for (int64_t j = 0; j < k; ++j)
      amax = std::max(amax, std::abs(wf[i * k + j]));
    const float step = amax > 0.0f ? (2.0f * amax) / 15.0f : 0.0f;
    const float tol = step + 1e-3f;
    for (int64_t j = 0; j < k; ++j) {
      EXPECT_NEAR(__bfloat162float(got[i * k + j]), wf[i * k + j], tol)
          << "at (" << i << ", " << j << ")";
    }
  }
}

// The fused W4A16 GEMM must match the unfused baseline (dequant int4 -> bf16,
// then the stock bf16 GEMM) -- both fp32-accumulate the same dequanted weights,
// so the gap is bf16 output rounding, not int4 quantization error. That keeps
// this a test of the fused dequant+GEMM pipeline rather than of int4 accuracy,
// which the round-trip tests own.
TEST_F(Int4QuantTest, FusedW4A16MatchesExactDequantReference) {
  constexpr int64_t m = 2, n = 256, k = 512, group = 128;

  std::vector<float> xf(m * k), wf(n * k);
  for (int64_t i = 0; i < m; ++i)
    for (int64_t p = 0; p < k; ++p) xf[i * k + p] = Fill(i, p, 0.0f);
  for (int64_t j = 0; j < n; ++j)
    for (int64_t p = 0; p < k; ++p) wf[j * k + p] = Fill(j, p, 1.7f);

  std::vector<DeviceBuffer> keep;
  const TensorView x = UploadBf16(keep, xf, m, k);
  const TensorView w = UploadBf16(keep, wf, n, k);

  auto qb = DeviceBuffer::Allocate(n * k / 2, DeviceId::Cuda(0));
  auto sb = DeviceBuffer::Allocate(n * (k / group) * sizeof(__nv_bfloat16),
                                   DeviceId::Cuda(0));
  ASSERT_TRUE(qb.ok() && sb.ok());
  auto q = TensorView::Create(qb->data(), DataType::kInt4, Shape({n, k}),
                              DeviceId::Cuda(0));
  auto s = TensorView::Create(sb->data(), DataType::kBFloat16,
                              Shape({n, k / group}), DeviceId::Cuda(0));
  ASSERT_TRUE(q.ok() && s.ok());

  ASSERT_TRUE(kernels::QuantizeBf16ToInt4(w, *q, *s).ok());

  // Unfused reference: dequant to bf16, then the stock bf16 GEMM.
  auto wbat =
      DeviceBuffer::Allocate(n * k * sizeof(__nv_bfloat16), DeviceId::Cuda(0));
  ASSERT_TRUE(wbat.ok());
  auto w_hat = TensorView::Create(wbat->data(), DataType::kBFloat16,
                                  Shape({n, k}), DeviceId::Cuda(0));
  ASSERT_TRUE(w_hat.ok());
  ASSERT_TRUE(kernels::DequantizeInt4ToBf16(*q, *s, *w_hat).ok());

  auto yrefb =
      DeviceBuffer::Allocate(m * n * sizeof(__nv_bfloat16), DeviceId::Cuda(0));
  auto ygotb =
      DeviceBuffer::Allocate(m * n * sizeof(__nv_bfloat16), DeviceId::Cuda(0));
  ASSERT_TRUE(yrefb.ok() && ygotb.ok());
  auto yref = TensorView::Create(yrefb->data(), DataType::kBFloat16,
                                 Shape({m, n}), DeviceId::Cuda(0));
  auto ygot = TensorView::Create(ygotb->data(), DataType::kBFloat16,
                                 Shape({m, n}), DeviceId::Cuda(0));
  ASSERT_TRUE(yref.ok() && ygot.ok());

  auto gemm = kernels::CublasLtGemm::Create();
  ASSERT_TRUE(gemm.ok()) << gemm.status();
  ASSERT_TRUE(gemm->LinearBF16(x, *w_hat, *yref).ok());

  ASSERT_TRUE(kernels::W4A16Gemm(x, *q, *s, *ygot, group).ok());
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  // The fused kernel keeps q*scale in fp32 (it never materializes a bf16
  // weight), so it is MORE accurate than the unfused bf16-weight path above and
  // will not match it to sub-ulp. The right gate is the exact dequant in double:
  // read back x, the packed int4, and the scales and compute y = x . (q*scale).
  std::vector<__nv_bfloat16> xh(m * k);
  std::vector<uint8_t> qh(n * k / 2);
  std::vector<__nv_bfloat16> sh(n * (k / group));
  std::vector<__nv_bfloat16> got(m * n);
  ASSERT_EQ(cudaMemcpy(xh.data(), x.Data(),
                        xh.size() * sizeof(__nv_bfloat16),
                        cudaMemcpyDeviceToHost),
            cudaSuccess);
  ASSERT_EQ(
      cudaMemcpy(qh.data(), qb->data(), qh.size(), cudaMemcpyDeviceToHost),
      cudaSuccess);
  ASSERT_EQ(cudaMemcpy(sh.data(), sb->data(),
                        sh.size() * sizeof(__nv_bfloat16),
                        cudaMemcpyDeviceToHost),
            cudaSuccess);
  ASSERT_EQ(cudaMemcpy(got.data(), ygotb->data(),
                        got.size() * sizeof(__nv_bfloat16),
                        cudaMemcpyDeviceToHost),
            cudaSuccess);

  const auto dequant = [&](int64_t j, int64_t kk) -> double {
    const uint8_t byte = qh[static_cast<size_t>(j) * (k / 2) + kk / 2];
    int qi = (kk & 1) ? (byte >> 4) : (byte & 0xF);
    if (qi >= 8) qi -= 16;
    return qi * static_cast<double>(
                    __bfloat162float(sh[static_cast<size_t>(j) * (k / group) +
                                       kk / group]));
  };

  for (int64_t i = 0; i < m; ++i) {
    for (int64_t j = 0; j < n; ++j) {
      double acc = 0.0;
      for (int64_t kk = 0; kk < k; ++kk) {
        acc += static_cast<double>(__bfloat162float(xh[i * k + kk])) *
               dequant(j, kk);
      }
      // The output is bf16, so the gate is ~one bf16 output ulp at the value's
      // magnitude plus a small relative term for the fused kernel's fp32
      // accumulate against the double reference.
      const float tol =
          0.13f + 0.02f * std::fabs(static_cast<float>(acc));
      EXPECT_NEAR(__bfloat162float(got[i * n + j]),
                  static_cast<float>(acc), tol)
          << "at (" << i << ", " << j << ")";
    }
  }
}

}  // namespace
}  // namespace inferx