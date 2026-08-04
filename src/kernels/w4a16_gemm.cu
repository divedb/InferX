#include "inferx/kernels/w4a16_gemm.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "inferx/core/cuda_utils.h"

namespace inferx::kernels {
namespace {

using bf16 = __nv_bfloat16;

// Output columns per block, and the block's thread count: one thread owns one
// output column and walks the full contraction. That puts a thread's weight
// reads at consecutive k offsets (cache- and L2-friendly within the thread); the
// cross-thread stride is the thing a tensor-core / warp-shuffle version would
// fix, and is what the "tune next" step attacks. This first cut targets
// correctness and the bandwidth-bound decode regime, not peak throughput.
constexpr int kBN = 128;

__global__ void W4A16GemmKernel(const bf16* __restrict__ x,
                                const uint8_t* __restrict__ w,
                                const bf16* __restrict__ scales,
                                bf16* __restrict__ y, int m, int n, int k,
                                int group) {
  const int j = blockIdx.x * kBN + threadIdx.x;
  if (j >= n) return;

  const int groups_per_row = k / group;
  const uint8_t* wr = w + static_cast<int64_t>(j) * (k / 2);
  const bf16* ws = scales + static_cast<int64_t>(j) * groups_per_row;

  for (int i = 0; i < m; ++i) {
    const bf16* xr = x + static_cast<int64_t>(i) * k;
    float acc = 0.0f;
    for (int kk = 0; kk < k; ++kk) {
      const uint8_t byte = wr[kk >> 1];
      // Low nibble is element 2*b, high nibble is 2*b+1 (see QuantizeBf16ToInt4).
      int q = (kk & 1) ? (byte >> 4) : (byte & 0xF);
      if (q >= 8) q -= 16;  // sign-extend the two's-complement nibble
      acc += __bfloat162float(xr[kk]) * q *
             __bfloat162float(ws[kk / group]);
    }
    y[static_cast<int64_t>(i) * n + j] = __float2bfloat16(acc);
  }
}

}  // namespace

Status W4A16Gemm(const TensorView& x, const TensorView& w_int4,
                 const TensorView& scales, const TensorView& y, int64_t group,
                 cudaStream_t stream) {
  if (!x.IsDefined() || !x.IsCuda()) {
    return InvalidArgumentError("W4A16Gemm: x must be a defined CUDA tensor");
  }
  if (x.GetDataType() != DataType::kBFloat16) {
    return InvalidArgumentError("W4A16Gemm: x is ", DataTypeName(x.GetDataType()),
                                ", expected bf16");
  }
  if (x.Rank() != 2) {
    return InvalidArgumentError("W4A16Gemm: x must be rank 2");
  }
  if (w_int4.GetDataType() != DataType::kInt4) {
    return InvalidArgumentError("W4A16Gemm: w is ", DataTypeName(w_int4.GetDataType()),
                                ", expected int4");
  }
  if (w_int4.Rank() != 2) {
    return InvalidArgumentError("W4A16Gemm: w must be rank 2");
  }
  if (scales.GetDataType() != DataType::kBFloat16) {
    return InvalidArgumentError("W4A16Gemm: scales is ",
                                DataTypeName(scales.GetDataType()),
                                ", expected bf16");
  }

  const int64_t m = x.Dim(0);
  const int64_t k = x.Dim(1);
  const int64_t n = w_int4.Dim(0);

  if (w_int4.Dim(1) != k) {
    return InvalidArgumentError("W4A16Gemm: w is [", w_int4.Dim(0), ", ",
                                w_int4.Dim(1), "], expected [", n, ", ", k, "]");
  }
  if (y.Rank() != 2 || y.Dim(0) != m || y.Dim(1) != n) {
    return InvalidArgumentError("W4A16Gemm: y is [", y.Dim(0), ", ", y.Dim(1),
                                "], expected [", m, ", ", n, "]");
  }
  if (group <= 0 || k % group != 0) {
    return InvalidArgumentError("W4A16Gemm: group ", group,
                                " must divide k=", k);
  }
  if (scales.Rank() != 2 || scales.Dim(0) != n ||
      scales.Dim(1) != k / group) {
    return InvalidArgumentError("W4A16Gemm: scales is [", scales.Dim(0), ", ",
                                scales.Dim(1), "], expected [", n, ", ",
                                k / group, "]");
  }
  if (k % 2 != 0) {
    return InvalidArgumentError("W4A16Gemm: k must be even, got ", k);
  }

  if (m == 0 || n == 0 || k == 0) return OkStatus();

  const int blocks = static_cast<int>((n + kBN - 1) / kBN);
  W4A16GemmKernel<<<blocks, kBN, 0, stream>>>(
      static_cast<const bf16*>(x.Data()),
      static_cast<const uint8_t*>(w_int4.Data()),
      static_cast<const bf16*>(scales.Data()),
      static_cast<bf16*>(y.Data()), static_cast<int>(m), static_cast<int>(n),
      static_cast<int>(k), static_cast<int>(group));

  INFERX_CUDA_RETURN_IF_ERROR(cudaGetLastError());
  return OkStatus();
}

}  // namespace inferx::kernels
