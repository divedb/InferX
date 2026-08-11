#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <type_traits>

#include "inferx/backends/cuda/cuda_utils.h"
#include "inferx/ops/mxfp4_gemm.h"

namespace inferx::ops {
namespace {

using bf16 = __nv_bfloat16;

// FP4 E2M1 in sign-magnitude nibble order. Decode arithmetically rather than
// indexing constant memory: a warp's nibbles are data-dependent and divergent
// constant addresses serialize into as many as 16 transactions.
__device__ __forceinline__ float DecodeFp4(uint8_t nibble) {
  const int magnitude = nibble & 0x7;
  const int exponent = magnitude >> 1;
  const int mantissa = magnitude & 1;
  float value;
  if (exponent == 0) {
    value = 0.5f * static_cast<float>(mantissa);
  } else {
    const float power =
        __int_as_float(static_cast<uint32_t>(exponent + 126) << 23);
    value = power * (1.0f + 0.5f * static_cast<float>(mantissa));
  }
  return (nibble & 0x8) ? -value : value;
}

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
  // split, so even input rows land in the first half and odd rows in the
  // second.
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
      const float scale_factor =
          __int_as_float(static_cast<uint32_t>(e8m0) << 23);
      acc += __bfloat162float(xr[kk]) * DecodeFp4(nibble) * scale_factor;
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
// factor are computed once per lane per step. The activation rows live in
// shared memory (every column reuses them), and each lane keeps one fp32
// accumulator per row, so the weight row is read exactly once for the whole
// batch.
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
      const float wv = DecodeFp4(nibble) * scale_factor;
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
    const float wv = DecodeFp4(nibble) * scale_factor;
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

// One warp owns one output column for a chunk of grouped rows. The expert's
// begin/end stay in device memory; an overprovisioned row-chunk grid makes the
// launch shape independent of the routing result and therefore graph-safe.
template <int RowsPerChunk>
__global__ void Mxfp4GroupedKernel(const bf16* __restrict__ x,
                                   const int32_t* __restrict__ offsets,
                                   const uint8_t* __restrict__ blocks,
                                   const uint8_t* __restrict__ scales,
                                   const bf16* __restrict__ bias,
                                   bf16* __restrict__ y, int experts, int n,
                                   int k, bool deinterleave) {
  const int warp = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  const int j_raw = blockIdx.x * kWarpsPerBlock + warp;
  if (j_raw >= n) return;

  int j = j_raw;
  if (deinterleave) {
    const int half = n / 2;
    j = (j_raw & 1) ? half + (j_raw >> 1) : (j_raw >> 1);
  }
  int expert_first = 0;
  int expert_last = experts;
  if constexpr (RowsPerChunk == 1) {
    // Decode maps grid.y directly to one grouped assignment. Locate its expert
    // with upper_bound(offsets, row) rather than making every warp scan all E
    // ranges. Only lane 0 performs the five global reads for E=32.
    int found = 0;
    if (lane == 0) {
      int lo = 0;
      int hi = experts;
      const int grouped_row = static_cast<int>(blockIdx.y);
      while (lo < hi) {
        const int mid = (lo + hi) / 2;
        if (offsets[mid + 1] <= grouped_row)
          lo = mid + 1;
        else
          hi = mid;
      }
      found = lo;
    }
    expert_first = __shfl_sync(~0u, found, 0);
    expert_last = expert_first + 1;
  }

  // Prefill loops over experts inside a warp so a row chunk can reuse each
  // weight value across up to 16 routed rows. Decode takes the direct one-row
  // path above and executes this body exactly once.
  for (int expert = expert_first; expert < expert_last; ++expert) {
    const int begin = offsets[expert];
    const int end = offsets[expert + 1];
    const int row0 = RowsPerChunk == 1 ? static_cast<int>(blockIdx.y)
                                       : begin + blockIdx.y * RowsPerChunk;
    if (row0 >= end) continue;
    const int rows = min(RowsPerChunk, end - row0);
    const int64_t weight_row = static_cast<int64_t>(expert) * n + j_raw;
    const uint8_t* wr = blocks + weight_row * (k / 2);
    const uint8_t* sr = scales + weight_row * (k / 32);

    float acc[RowsPerChunk] = {};
    for (int base = 0; base + kVecStride <= k; base += kVecStride) {
      const int byte_off = base / 2 + lane * (kVecElems / 2);
      const uint32_t packed = *reinterpret_cast<const uint32_t*>(wr + byte_off);
      const int k0 = base + lane * kVecElems;
      const float scale =
          __int_as_float(static_cast<uint32_t>(sr[k0 >> 5]) << 23);
#pragma unroll
      for (int e = 0; e < kVecElems; ++e) {
        const uint8_t byte = (packed >> (8 * (e / 2))) & 0xff;
        const int nibble = (e & 1) ? (byte >> 4) : (byte & 0xf);
        const float wv = DecodeFp4(nibble) * scale;
        const int kk = k0 + e;
#pragma unroll
        for (int r = 0; r < RowsPerChunk; ++r) {
          if (r < rows)
            acc[r] +=
                __bfloat162float(x[static_cast<int64_t>(row0 + r) * k + kk]) *
                wv;
        }
      }
    }

    const int covered = (k / kVecStride) * kVecStride;
    for (int kk = covered + lane; kk < k; kk += 32) {
      const uint8_t byte = wr[kk >> 1];
      const int nibble = (kk & 1) ? (byte >> 4) : (byte & 0xf);
      const float scale =
          __int_as_float(static_cast<uint32_t>(sr[kk >> 5]) << 23);
      const float wv = DecodeFp4(nibble) * scale;
#pragma unroll
      for (int r = 0; r < RowsPerChunk; ++r) {
        if (r < rows)
          acc[r] +=
              __bfloat162float(x[static_cast<int64_t>(row0 + r) * k + kk]) * wv;
      }
    }

#pragma unroll
    for (int r = 0; r < RowsPerChunk; ++r) {
      if (r >= rows) break;
      float v = acc[r];
      for (int off = 16; off; off >>= 1) v += __shfl_xor_sync(~0u, v, off);
      if (lane == 0) {
        if (bias != nullptr)
          v += __bfloat162float(bias[static_cast<int64_t>(expert) * n + j]);
        y[static_cast<int64_t>(row0 + r) * n + j] = __float2bfloat16(v);
      }
    }
  }
}

}  // namespace

Status Mxfp4Gemm(const TensorView& x, const TensorView& blocks,
                 const TensorView& scales, const TensorView& y,
                 bool deinterleave, Stream stream) {
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
  if (!y.IsDefined() || y.GetDataType() != DataType::kBFloat16 ||
      y.Rank() != 2) {
    return InvalidArgumentError("Mxfp4Gemm: y must be a 2-D bf16 tensor");
  }

  const int64_t m = x.Dim(0);
  const int64_t k = x.Dim(1);
  const int64_t n = blocks.Dim(0);

  if (blocks.Dim(1) != k / 2) {
    return InvalidArgumentError("Mxfp4Gemm: blocks is [", blocks.Dim(0), ", ",
                                blocks.Dim(1),
                                "], expected column count k/2 = ", k / 2);
  }
  if (scales.Dim(0) != n || scales.Dim(1) != k / 32) {
    return InvalidArgumentError("Mxfp4Gemm: scales is [", scales.Dim(0), ", ",
                                scales.Dim(1), "], expected [", n, ", ", k / 32,
                                "]");
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
      m <= kMaxRows &&
      static_cast<int64_t>(m) * k * sizeof(bf16) <= kMaxSmemBytes;

  if (vector_ok) {
    const int blks =
        static_cast<int>((n + kWarpsPerBlock - 1) / kWarpsPerBlock);
    Mxfp4GemmVectorKernel<<<blks, kWarpsPerBlock * 32,
                            static_cast<size_t>(m) * static_cast<size_t>(k) *
                                sizeof(bf16),
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

Status Mxfp4GroupedGemm(const TensorView& x, const TensorView& offsets,
                        const TensorView& blocks, const TensorView& scales,
                        const TensorView& bias, const TensorView& y,
                        bool deinterleave, Stream stream) {
  if (!x.IsDefined() || !x.IsCuda() || x.GetDataType() != DataType::kBFloat16 ||
      x.Rank() != 2)
    return InvalidArgumentError("Mxfp4GroupedGemm: x must be 2-D bf16 CUDA");
  if (!offsets.IsDefined() || !offsets.IsCuda() ||
      offsets.GetDataType() != DataType::kInt32 || offsets.Rank() != 1)
    return InvalidArgumentError(
        "Mxfp4GroupedGemm: offsets must be 1-D int32 CUDA");
  if (!blocks.IsDefined() || !blocks.IsCuda() ||
      blocks.GetDataType() != DataType::kUInt8 ||
      (blocks.Rank() != 3 && blocks.Rank() != 4) || !scales.IsDefined() ||
      !scales.IsCuda() || scales.GetDataType() != DataType::kUInt8 ||
      scales.Rank() != 3)
    return InvalidArgumentError(
        "Mxfp4GroupedGemm: blocks must be 3-D/4-D and scales 3-D u8 CUDA");
  const int64_t assignments = x.Dim(0), k = x.Dim(1);
  const int64_t experts = blocks.Dim(0), n = blocks.Dim(1);
  const int64_t packed_k =
      blocks.Rank() == 3 ? blocks.Dim(2) : blocks.Dim(2) * blocks.Dim(3);
  if (offsets.Dim(0) != experts + 1 || packed_k != k / 2 ||
      scales.Dim(0) != experts || scales.Dim(1) != n ||
      scales.Dim(2) != k / 32 || k % 32 != 0)
    return InvalidArgumentError(
        "Mxfp4GroupedGemm: incompatible weight/offset shapes");
  if (!y.IsDefined() || !y.IsCuda() || y.GetDataType() != DataType::kBFloat16 ||
      y.Rank() != 2 || y.Dim(0) != assignments || y.Dim(1) != n)
    return InvalidArgumentError("Mxfp4GroupedGemm: y has incompatible shape");
  if (bias.IsDefined() &&
      (!bias.IsCuda() || bias.GetDataType() != DataType::kBFloat16 ||
       bias.Rank() != 2 || bias.Dim(0) != experts || bias.Dim(1) != n))
    return InvalidArgumentError(
        "Mxfp4GroupedGemm: bias has incompatible shape");
  if (assignments == 0 || n == 0) return OkStatus();

  const unsigned grid_x =
      static_cast<unsigned>((n + kWarpsPerBlock - 1) / kWarpsPerBlock);
  const auto launch = [&](auto rows_tag) {
    constexpr int rows = decltype(rows_tag)::value;
    const dim3 grid(grid_x,
                    static_cast<unsigned>((assignments + rows - 1) / rows), 1);
    Mxfp4GroupedKernel<rows><<<grid, kWarpsPerBlock * 32, 0, stream>>>(
        static_cast<const bf16*>(x.Data()),
        static_cast<const int32_t*>(offsets.Data()),
        static_cast<const uint8_t*>(blocks.Data()),
        static_cast<const uint8_t*>(scales.Data()),
        bias.IsDefined() ? static_cast<const bf16*>(bias.Data()) : nullptr,
        static_cast<bf16*>(y.Data()), static_cast<int>(experts),
        static_cast<int>(n), static_cast<int>(k), deinterleave);
  };
  // Decode has four total assignments (top-4 for one token), and each expert
  // normally owns one. A single accumulator avoids paying the registers and
  // predicated FMA loop for 16 rows. Larger batches retain weight reuse across
  // up to 16 routed rows.
  if (assignments <= 4) {
    launch(std::integral_constant<int, 1>{});
  } else {
    launch(std::integral_constant<int, kMaxRows>{});
  }
  INFERX_CUDA_RETURN_IF_ERROR(cudaGetLastError());
  return OkStatus();
}

}  // namespace inferx::ops
