#include "inferx/kernels/w4a16_gemm.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "inferx/core/cuda_utils.h"

namespace inferx::kernels {
namespace {

using bf16 = __nv_bfloat16;

// ============================== scalar fallback =============================
// One thread per output column, scalar loop over k. Correct for any shape; the
// fallback for shapes the vector kernel below does not serve (large m*k, where
// the activation no longer fits in shared memory). Not bandwidth-tuned.
constexpr int kScalarBN = 128;

__global__ void W4A16GemmScalarKernel(const bf16* __restrict__ x,
                                      const uint8_t* __restrict__ w,
                                      const bf16* __restrict__ scales,
                                      bf16* __restrict__ y, int m, int n, int k,
                                      int group) {
  const int j = blockIdx.x * kScalarBN + threadIdx.x;
  if (j >= n) return;

  const int groups_per_row = k / group;
  const uint8_t* wr = w + static_cast<int64_t>(j) * (k / 2);
  const bf16* ws = scales + static_cast<int64_t>(j) * groups_per_row;

  for (int i = 0; i < m; ++i) {
    const bf16* xr = x + static_cast<int64_t>(i) * k;
    float acc = 0.0f;
    for (int kk = 0; kk < k; ++kk) {
      const uint8_t byte = wr[kk >> 1];
      int q = (kk & 1) ? (byte >> 4) : (byte & 0xF);
      if (q >= 8) q -= 16;
      acc += __bfloat162float(xr[kk]) * q *
             __bfloat162float(ws[kk / group]);
    }
    y[static_cast<int64_t>(i) * n + j] = __float2bfloat16(acc);
  }
}

// ============================== vector decode ===============================
// One warp owns one output column j and computes y[0..m-1, j]. The 32 lanes
// stride over the contraction dim, reading the int4 weight row as coalesced
// 128-byte transactions -- one uint32 (8 nibbles) per lane per step -- which is
// the load pattern that lets the kernel approach the int4 bandwidth floor. The
// activation rows live in shared memory (every column reuses them), and each
// lane keeps one fp32 accumulator per row, so the weight row is read exactly
// once for the whole batch. That is the structural difference from the unfused
// dequant-then-bf16-GEMM baseline, and the reason W4A16 can beat bf16 on
// weight-bandwidth-bound decode.
constexpr int kWarpsPerBlock = 4;       // output columns per block
constexpr int kVecElems = 8;            // one uint32 load = 8 int4 elements
constexpr int kVecStride = 32 * kVecElems;  // elements per warp per step (256)
constexpr int kMaxRows = 16;            // accumulators per lane; decode batch

__global__ void W4A16GemmVectorKernel(const bf16* __restrict__ x,
                                      const uint8_t* __restrict__ w,
                                      const bf16* __restrict__ scales,
                                      bf16* __restrict__ y, int m, int n, int k,
                                      int group) {
  const int warp = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  const int j = blockIdx.x * kWarpsPerBlock + warp;

  extern __shared__ bf16 xs[];  // [m, k]

  for (int t = threadIdx.x; t < m * k; t += kWarpsPerBlock * 32) xs[t] = x[t];
  __syncthreads();

  if (j >= n) return;

  const uint8_t* wr = w + static_cast<int64_t>(j) * (k / 2);
  const bf16* ws = scales + static_cast<int64_t>(j) * (k / group);

  float acc[kMaxRows];
#pragma unroll
  for (int i = 0; i < kMaxRows; ++i) acc[i] = 0.0f;

  // Vectorized main loop: 256 elements per warp per step, coalesced.
  for (int base = 0; base + kVecStride <= k; base += kVecStride) {
    const int byte_off = base / 2 + lane * (kVecElems / 2);  // 4-byte aligned
    const uint32_t packed =
        *reinterpret_cast<const uint32_t*>(wr + byte_off);
    const int k0 = base + lane * kVecElems;
#pragma unroll
    for (int e = 0; e < kVecElems; ++e) {
      const uint8_t byte = (packed >> (8 * (e / 2))) & 0xFF;
      int q = (e & 1) ? (byte >> 4) : (byte & 0xF);
      if (q >= 8) q -= 16;
      const int kk = k0 + e;
      const float wv = q * __bfloat162float(ws[kk / group]);
#pragma unroll
      for (int i = 0; i < kMaxRows; ++i) {
        if (i < m) acc[i] += __bfloat162float(xs[(int64_t)i * k + kk]) * wv;
      }
    }
  }

  // Scalar tail for k not divisible by kVecStride.
  const int covered = (k / kVecStride) * kVecStride;
  for (int kk = covered + lane; kk < k; kk += 32) {
    const uint8_t byte = wr[kk >> 1];
    int q = (kk & 1) ? (byte >> 4) : (byte & 0xF);
    if (q >= 8) q -= 16;
    const float wv = q * __bfloat162float(ws[kk / group]);
#pragma unroll
    for (int i = 0; i < kMaxRows; ++i) {
      if (i < m) acc[i] += __bfloat162float(xs[(int64_t)i * k + kk]) * wv;
    }
  }

  // One warp reduce per row; lane 0 writes the output.
#pragma unroll
  for (int i = 0; i < m; ++i) {
    float v = acc[i];
    for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(~0u, v, off);
    if (lane == 0) y[(int64_t)i * n + j] = __float2bfloat16(v);
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

  // The vector kernel needs the activation rows to fit in shared memory and a
  // bounded number of rows (one accumulator per lane per row). Decode batch
  // sizes satisfy both; larger prefill batches fall through to the scalar
  // kernel, which is correct for any shape. The scalar path is also the
  // reference the vector path is diffed against.
  constexpr int64_t kMaxSmemBytes = 48 * 1024;
  const bool vector_ok =
      m <= kMaxRows && static_cast<int64_t>(m) * k * sizeof(bf16) <= kMaxSmemBytes;

  if (vector_ok) {
    const int blocks = static_cast<int>((n + kWarpsPerBlock - 1) / kWarpsPerBlock);
    W4A16GemmVectorKernel<<<blocks, kWarpsPerBlock * 32,
                           static_cast<size_t>(m) * static_cast<size_t>(k) * sizeof(bf16),
                           stream>>>(
        static_cast<const bf16*>(x.Data()),
        static_cast<const uint8_t*>(w_int4.Data()),
        static_cast<const bf16*>(scales.Data()),
        static_cast<bf16*>(y.Data()), static_cast<int>(m), static_cast<int>(n),
        static_cast<int>(k), static_cast<int>(group));
  } else {
    const int blocks = static_cast<int>((n + kScalarBN - 1) / kScalarBN);
    W4A16GemmScalarKernel<<<blocks, kScalarBN, 0, stream>>>(
        static_cast<const bf16*>(x.Data()),
        static_cast<const uint8_t*>(w_int4.Data()),
        static_cast<const bf16*>(scales.Data()),
        static_cast<bf16*>(y.Data()), static_cast<int>(m), static_cast<int>(n),
        static_cast<int>(k), static_cast<int>(group));
  }

  INFERX_CUDA_RETURN_IF_ERROR(cudaGetLastError());
  return OkStatus();
}

}  // namespace inferx::kernels
