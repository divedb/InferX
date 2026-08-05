#include "inferx/kernels/mxfp4_gemm.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "inferx/core/cuda_utils.h"

namespace inferx::kernels {
namespace {

using bf16 = __nv_bfloat16;

// FP4 E2M1 in sign-magnitude nibble order -- the same table mxfp4.cu uses.
// Indexed by the raw 4-bit nibble value (low nibble of each byte first).
__device__ __constant__ float kFp4Lut[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f};

// ============================== scalar fallback =============================
// One thread per output column, scalar loop over k. Correct for any shape;
// the fallback for shapes the vector kernel below does not serve (large m*k,
// where the activation no longer fits in shared memory).
constexpr int kScalarBN = 128;

__global__ void Mxfp4GemmScalarKernel(const bf16* __restrict__ x,
                                      const uint8_t* __restrict__ blocks,
                                      const uint8_t* __restrict__ scales,
                                      bf16* __restrict__ y, int m, int n, int k,
                                      bool deinterleave) {
  const int j_raw = blockIdx.x * kScalarBN + threadIdx.x;
  if (j_raw >= n) return;

  // Output column: the checkpoint alternates gate/up rows for gate_up weights
  // (row 2i = gate_i, row 2i+1 = up_i). The activation expects [gate | up]
  // split, so even input rows land in the first half and odd rows in the second.
  int j = j_raw;
  if (deinterleave) {
    const int half = n / 2;
    j = (j_raw & 1) ? half + (j_raw >> 1) : (j_raw >> 1);
  }

  const int groups_per_row = k / 32;
  const uint8_t* wr = blocks + static_cast<int64_t>(j_raw) * (k / 2);
  const uint8_t* sr = scales + static_cast<int64_t>(j_raw) * groups_per_row;

  for (int i = 0; i < m; ++i) {
    const bf16* xr = x + static_cast<int64_t>(i) * k;
    float acc = 0.0f;
    for (int kk = 0; kk < k; ++kk) {
      const uint8_t byte = wr[kk >> 1];
      const int nibble = (kk & 1) ? (byte >> 4) : (byte & 0xF);
      const uint8_t e8m0 = sr[kk >> 5];  // kk / 32
      const float scale_factor = __int_as_float(static_cast<uint32_t>(e8m0) << 23);
      acc += __bfloat162float(xr[kk]) * kFp4Lut[nibble] * scale_factor;
    }
    y[static_cast<int64_t>(i) * n + j] = __float2bfloat16(acc);
  }
}

// ============================== vector decode ===============================
// One warp owns one output column j_raw and computes y[0..m-1, remap(j_raw)].
// The 32 lanes stride over the contraction dim, reading the MXFP4 weight row as
// coalesced 128-byte transactions -- one uint32 (8 nibbles) per lane per step.
// Each lane's 8 elements share one E8M0 group scale (8 < 32 and the lane stride
// keeps them off group boundaries), so the scale lookup and the fp32 scale
// factor are computed once per lane per step. The activation rows live in shared
// memory (every column reuses them), and each lane keeps one fp32 accumulator
// per row, so the weight row is read exactly once for the whole batch.
constexpr int kWarpsPerBlock = 4;
constexpr int kVecElems = 8;
constexpr int kVecStride = 32 * kVecElems;  // 256
constexpr int kMaxRows = 16;

__global__ void Mxfp4GemmVectorKernel(const bf16* __restrict__ x,
                                      const uint8_t* __restrict__ blocks,
                                      const uint8_t* __restrict__ scales,
                                      bf16* __restrict__ y, int m, int n, int k,
                                      bool deinterleave) {
  const int warp = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  const int j_raw = blockIdx.x * kWarpsPerBlock + warp;

  extern __shared__ bf16 xs[];  // [m, k]

  for (int t = threadIdx.x; t < m * k; t += kWarpsPerBlock * 32) xs[t] = x[t];
  __syncthreads();

  if (j_raw >= n) return;

  int j = j_raw;
  if (deinterleave) {
    const int half = n / 2;
    j = (j_raw & 1) ? half + (j_raw >> 1) : (j_raw >> 1);
  }

  const int groups_per_row = k / 32;
  const uint8_t* wr = blocks + static_cast<int64_t>(j_raw) * (k / 2);
  const uint8_t* sr = scales + static_cast<int64_t>(j_raw) * groups_per_row;

  float acc[kMaxRows];
#pragma unroll
  for (int i = 0; i < kMaxRows; ++i) acc[i] = 0.0f;

  for (int base = 0; base + kVecStride <= k; base += kVecStride) {
    const int byte_off = base / 2 + lane * (kVecElems / 2);
    const uint32_t packed = *reinterpret_cast<const uint32_t*>(wr + byte_off);
    const int k0 = base + lane * kVecElems;

    // All 8 elements in this lane are within one E8M0 group (8 < 32, and the
    // lane stride of 8 keeps them off a group boundary: k0 % 32 is in
    // {0, 8, 16, 24}, never in {25..31}). One scale, one factor.
    const uint8_t e8m0 = sr[k0 >> 5];
    const float scale_factor =
        __int_as_float(static_cast<uint32_t>(e8m0) << 23);

#pragma unroll
    for (int e = 0; e < kVecElems; ++e) {
      const uint8_t byte = (packed >> (8 * (e / 2))) & 0xFF;
      const int nibble = (e & 1) ? (byte >> 4) : (byte & 0xF);
      const float wv = kFp4Lut[nibble] * scale_factor;
      const int kk = k0 + e;
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
    const int nibble = (kk & 1) ? (byte >> 4) : (byte & 0xF);
    const uint8_t e8m0 = sr[kk >> 5];
    const float scale_factor =
        __int_as_float(static_cast<uint32_t>(e8m0) << 23);
    const float wv = kFp4Lut[nibble] * scale_factor;
#pragma unroll
    for (int i = 0; i < kMaxRows; ++i) {
      if (i < m) acc[i] += __bfloat162float(xs[(int64_t)i * k + kk]) * wv;
    }
  }

#pragma unroll
  for (int i = 0; i < m; ++i) {
    float v = acc[i];
    for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(~0u, v, off);
    if (lane == 0) y[(int64_t)i * n + j] = __float2bfloat16(v);
  }
}

}  // namespace

Status Mxfp4Gemm(const TensorView& x, const TensorView& blocks,
                 const TensorView& scales, const TensorView& y,
                 bool deinterleave, cudaStream_t stream) {
  if (!x.IsDefined() || !x.IsCuda() || x.GetDataType() != DataType::kBFloat16 ||
      x.Rank() != 2) {
    return InvalidArgumentError("Mxfp4Gemm: x must be a 2-D bf16 CUDA tensor");
  }
  if (!blocks.IsDefined() || blocks.GetDataType() != DataType::kUInt8 ||
      blocks.Rank() != 2) {
    return InvalidArgumentError("Mxfp4Gemm: blocks must be a 2-D u8 tensor");
  }
  if (!scales.IsDefined() || scales.GetDataType() != DataType::kUInt8 ||
      scales.Rank() != 2) {
    return InvalidArgumentError("Mxfp4Gemm: scales must be a 2-D u8 tensor");
  }
  if (!y.IsDefined() || y.GetDataType() != DataType::kBFloat16 || y.Rank() != 2) {
    return InvalidArgumentError("Mxfp4Gemm: y must be a 2-D bf16 tensor");
  }

  const int64_t m = x.Dim(0);
  const int64_t k = x.Dim(1);
  const int64_t n = blocks.Dim(0);

  if (blocks.Dim(1) != k / 2) {
    return InvalidArgumentError("Mxfp4Gemm: blocks is [", blocks.Dim(0), ", ",
                                blocks.Dim(1), "], expected column count k/2 = ",
                                k / 2);
  }
  if (scales.Dim(0) != n || scales.Dim(1) != k / 32) {
    return InvalidArgumentError("Mxfp4Gemm: scales is [", scales.Dim(0), ", ",
                                scales.Dim(1), "], expected [", n, ", ",
                                k / 32, "]");
  }
  if (y.Dim(0) != m || y.Dim(1) != n) {
    return InvalidArgumentError("Mxfp4Gemm: y is [", y.Dim(0), ", ", y.Dim(1),
                                "], expected [", m, ", ", n, "]");
  }
  if (k % 32 != 0) {
    return InvalidArgumentError("Mxfp4Gemm: k must be a multiple of 32, got ",
                                k);
  }

  if (m == 0 || n == 0 || k == 0) return OkStatus();

  constexpr int64_t kMaxSmemBytes = 48 * 1024;
  const bool vector_ok =
      m <= kMaxRows && static_cast<int64_t>(m) * k * sizeof(bf16) <= kMaxSmemBytes;

  if (vector_ok) {
    const int blks = static_cast<int>((n + kWarpsPerBlock - 1) / kWarpsPerBlock);
    Mxfp4GemmVectorKernel<<<blks, kWarpsPerBlock * 32,
                            static_cast<size_t>(m) * static_cast<size_t>(k) * sizeof(bf16),
                            stream>>>(
        static_cast<const bf16*>(x.Data()),
        static_cast<const uint8_t*>(blocks.Data()),
        static_cast<const uint8_t*>(scales.Data()),
        static_cast<bf16*>(y.Data()), static_cast<int>(m), static_cast<int>(n),
        static_cast<int>(k), deinterleave);
  } else {
    const int blks = static_cast<int>((n + kScalarBN - 1) / kScalarBN);
    Mxfp4GemmScalarKernel<<<blks, kScalarBN, 0, stream>>>(
        static_cast<const bf16*>(x.Data()),
        static_cast<const uint8_t*>(blocks.Data()),
        static_cast<const uint8_t*>(scales.Data()),
        static_cast<bf16*>(y.Data()), static_cast<int>(m), static_cast<int>(n),
        static_cast<int>(k), deinterleave);
  }

  INFERX_CUDA_RETURN_IF_ERROR(cudaGetLastError());
  return OkStatus();
}

}  // namespace inferx::kernels
