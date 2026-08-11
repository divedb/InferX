#include "inferx/kernels/quantize.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include "inferx/core/cuda_utils.h"

namespace inferx::kernels {
namespace {

using bf16 = __nv_bfloat16;

constexpr int kBlock = 256;

// Widens either half format to fp32, so one copy of each kernel below serves
// both -- the W4A16 bench is f16, the model's activations are bf16.
__device__ inline float Widen(__half x) { return __half2float(x); }
__device__ inline float Widen(__nv_bfloat16 x) { return __bfloat162float(x); }

// Grid-stride amax with a block reduction, finished with one atomic per block.
//
// atomicMax on the *bit pattern* rather than on the float: IEEE-754 orders
// non-negative floats identically to their unsigned bit patterns, and we only
// ever feed this fabsf() output, so the integer maximum is the float maximum.
// CUDA has no atomicMax for float, and doing this with a CAS loop would be
// slower for no gain.
template <typename Src>
__global__ void AbsMaxKernel(const Src* __restrict__ src, int64_t n,
                             float* __restrict__ out) {
  __shared__ float tile[kBlock];

  float local = 0.0f;

  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < n; i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    local = fmaxf(local, fabsf(Widen(src[i])));
  }

  tile[threadIdx.x] = local;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      tile[threadIdx.x] = fmaxf(tile[threadIdx.x], tile[threadIdx.x + stride]);
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    atomicMax(reinterpret_cast<unsigned int*>(out), __float_as_uint(tile[0]));
  }
}

// Turns the accumulated amax into the dequantization scale, in place.
__global__ void FinalizeScaleKernel(float* __restrict__ scale) {
  const float amax = *scale;

  // An all-zero tensor has no meaningful scale, and 0 would make the quantize
  // kernel divide by zero. 1 leaves the (all-zero) values unchanged.
  *scale = amax > 0.0f ? amax / kFloat8E4M3Max : 1.0f;
}

template <typename Src>
__global__ void QuantizeKernel(const Src* __restrict__ src,
                               __nv_fp8_storage_t* __restrict__ dst, int64_t n,
                               const float* __restrict__ scale) {
  // Read once per thread rather than per element: it is the same value for the
  // whole launch and the compiler cannot know that through a pointer.
  const float inv = 1.0f / *scale;

  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < n; i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    // __NV_SATFINITE clamps to ±448 instead of producing inf/NaN, so a scale
    // that turns out too small costs accuracy rather than correctness.
    dst[i] = __nv_cvt_float_to_fp8(Widen(src[i]) * inv, __NV_SATFINITE,
                                   __NV_E4M3);
  }
}

// Enough blocks to fill the device without launching one per element for a
// tensor with hundreds of millions of them.
int GridFor(int64_t n) {
  const int64_t want = (n + kBlock - 1) / kBlock;
  return static_cast<int>(want < 1024 ? (want < 1 ? 1 : want) : 1024);
}

Status CheckDeviceTensor(const TensorView& t, DataType expected,
                         const char* name) {
  if (!t.IsDefined()) return InvalidArgumentError(name, " is undefined");

  if (!t.IsCuda()) {
    return InvalidArgumentError(name, " is on ", t.Device().ToString(),
                                ", not a CUDA device");
  }

  if (t.GetDataType() != expected) {
    return InvalidArgumentError(name, " is ", DataTypeName(t.GetDataType()),
                                ", expected ", DataTypeName(expected));
  }

  return OkStatus();
}

// One block, two passes, no global synchronization: find the maximum magnitude
// across the whole tensor, reduce it within the block, then quantize. Reading
// the input twice is cheaper than the three extra launches the split version
// costs, because at decode sizes everything is in L2 by the second pass.
//
// One block also means one SM. That is the right trade for a decode
// activation -- a few thousand elements, where launch count dominates -- and
// the wrong one by an order of magnitude for a prefill activation, so
// QuantizeToF8E4M3Dynamic dispatches to the split path above that size.
template <typename Src>
__global__ void QuantizeDynamicKernel(const Src* __restrict__ src,
                                      __nv_fp8_storage_t* __restrict__ dst,
                                      int64_t n, float* __restrict__ scale_out) {
  __shared__ float tile[1024];

  float local = 0.0f;
  for (int64_t i = threadIdx.x; i < n; i += blockDim.x) {
    local = fmaxf(local, fabsf(Widen(src[i])));
  }

  tile[threadIdx.x] = local;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      tile[threadIdx.x] = fmaxf(tile[threadIdx.x], tile[threadIdx.x + stride]);
    }
    __syncthreads();
  }

  const float amax = tile[0];
  const float scale = amax > 0.0f ? amax / kFloat8E4M3Max : 1.0f;

  if (threadIdx.x == 0) *scale_out = scale;

  const float inv = 1.0f / scale;

  for (int64_t i = threadIdx.x; i < n; i += blockDim.x) {
    dst[i] = __nv_cvt_float_to_fp8(Widen(src[i]) * inv, __NV_SATFINITE,
                                   __NV_E4M3);
  }
}

}  // namespace

Status ComputeF8Scale(const TensorView& src, float* scale_dev,
                      Stream stream) {
  INFERX_RETURN_IF_ERROR(CheckDeviceTensor(src, DataType::kFloat16, "src"));

  if (scale_dev == nullptr) {
    return InvalidArgumentError("ComputeF8Scale: scale_dev is null");
  }

  const int64_t n = src.Numel();

  if (n == 0) return InvalidArgumentError("ComputeF8Scale: src is empty");

  // Zeroed rather than assumed: the kernel only ever raises this value, so a
  // stale maximum from a previous call would silently survive.
  INFERX_CUDA_RETURN_IF_ERROR(cudaMemsetAsync(scale_dev, 0, sizeof(float),
                                              stream));

  AbsMaxKernel<<<GridFor(n), kBlock, 0, stream>>>(
      static_cast<const __half*>(src.Data()), n, scale_dev);
  INFERX_CUDA_RETURN_IF_ERROR(cudaGetLastError());

  FinalizeScaleKernel<<<1, 1, 0, stream>>>(scale_dev);
  INFERX_CUDA_RETURN_IF_ERROR(cudaGetLastError());

  return OkStatus();
}

Status QuantizeF16ToF8E4M3(const TensorView& src, const TensorView& dst,
                           const float* scale_dev, Stream stream) {
  INFERX_RETURN_IF_ERROR(CheckDeviceTensor(src, DataType::kFloat16, "src"));
  INFERX_RETURN_IF_ERROR(
      CheckDeviceTensor(dst, DataType::kFloat8E4M3FN, "dst"));

  if (scale_dev == nullptr) {
    return InvalidArgumentError("QuantizeF16ToF8E4M3: scale_dev is null");
  }

  if (src.Numel() != dst.Numel()) {
    return InvalidArgumentError("QuantizeF16ToF8E4M3: src has ", src.Numel(),
                                " elements, dst has ", dst.Numel());
  }

  const int64_t n = src.Numel();

  if (n == 0) return OkStatus();

  QuantizeKernel<<<GridFor(n), kBlock, 0, stream>>>(
      static_cast<const __half*>(src.Data()),
      static_cast<__nv_fp8_storage_t*>(dst.Data()), n, scale_dev);
  INFERX_CUDA_RETURN_IF_ERROR(cudaGetLastError());

  return OkStatus();
}

Status QuantizeToF8E4M3Dynamic(const TensorView& src, const TensorView& dst,
                               float* scale_dev, Stream stream) {
  if (!src.IsDefined() || !src.IsCuda()) {
    return InvalidArgumentError("src must be a defined CUDA tensor");
  }
  if (dst.GetDataType() != DataType::kFloat8E4M3FN) {
    return InvalidArgumentError("dst is ", DataTypeName(dst.GetDataType()),
                                ", expected f8e4m3");
  }
  if (src.Numel() != dst.Numel()) {
    return InvalidArgumentError("src has ", src.Numel(), " elements, dst has ",
                                dst.Numel());
  }
  if (scale_dev == nullptr) {
    return InvalidArgumentError("scale_dev is null");
  }

  const int64_t n = src.Numel();
  if (n == 0) return OkStatus();

  if (src.GetDataType() != DataType::kBFloat16 &&
      src.GetDataType() != DataType::kFloat16) {
    return InvalidArgumentError("src is ", DataTypeName(src.GetDataType()),
                                ", expected f16 or bf16");
  }

  const bool bf16_src = src.GetDataType() == DataType::kBFloat16;
  auto* out = static_cast<__nv_fp8_storage_t*>(dst.Data());

  // Where the two strategies cross. Below it, the single-block kernel wins on
  // launch count: one launch instead of four, against a tensor small enough
  // that one SM reads it in microseconds. Above it, one SM is the bottleneck
  // and the extra launches disappear into the work -- 128k elements is a
  // 2048-token prefill activation at hidden 64, so every real prefill lands on
  // the split path and every decode activation lands on the single-block one.
  //
  // M10 is why this dispatch exists: serving with --fp8 measured 3.7k prefill
  // tok/s against bf16's 8.9k, and the GEMM microbenchmark could not see it
  // because it times the matmul on operands somebody else quantized. A whole
  // prefill's activations were going through one SM.
  constexpr int64_t kSplitPathElements = 128 * 1024;

  if (n < kSplitPathElements) {
    constexpr int kDynamicBlock = 1024;

    if (bf16_src) {
      QuantizeDynamicKernel<bf16><<<1, kDynamicBlock, 0, stream>>>(
          static_cast<const bf16*>(src.Data()), out, n, scale_dev);
    } else {
      QuantizeDynamicKernel<__half><<<1, kDynamicBlock, 0, stream>>>(
          static_cast<const __half*>(src.Data()), out, n, scale_dev);
    }

    INFERX_CUDA_RETURN_IF_ERROR(cudaGetLastError());
    return OkStatus();
  }

  // The split path: amax across the grid, then the scale, then the quantize.
  // Every step is stream-ordered and graph-capturable, including the memset --
  // AbsMaxKernel only ever raises the value, so a stale maximum from the
  // previous call would silently survive without it.
  INFERX_CUDA_RETURN_IF_ERROR(
      cudaMemsetAsync(scale_dev, 0, sizeof(float), stream));

  const int grid = GridFor(n);

  if (bf16_src) {
    AbsMaxKernel<bf16><<<grid, kBlock, 0, stream>>>(
        static_cast<const bf16*>(src.Data()), n, scale_dev);
  } else {
    AbsMaxKernel<__half><<<grid, kBlock, 0, stream>>>(
        static_cast<const __half*>(src.Data()), n, scale_dev);
  }
  INFERX_CUDA_RETURN_IF_ERROR(cudaGetLastError());

  FinalizeScaleKernel<<<1, 1, 0, stream>>>(scale_dev);
  INFERX_CUDA_RETURN_IF_ERROR(cudaGetLastError());

  if (bf16_src) {
    QuantizeKernel<bf16><<<grid, kBlock, 0, stream>>>(
        static_cast<const bf16*>(src.Data()), out, n, scale_dev);
  } else {
    QuantizeKernel<__half><<<grid, kBlock, 0, stream>>>(
        static_cast<const __half*>(src.Data()), out, n, scale_dev);
  }
  INFERX_CUDA_RETURN_IF_ERROR(cudaGetLastError());

  return OkStatus();
}

// ===========================================================================
// W4A16: per-group symmetric int4. See quantize.h for the format rationale.
// ===========================================================================

namespace {

// float -> f16/bf16. Widen() (defined above for the fp8 kernels) covers the
// reverse direction, so together the int4 kernels below are half-format
// agnostic: the W4A16 bench is f16, the model's weights/activations are bf16.
template <typename H>
__device__ __forceinline__ H Int4FromFloat(float x);
template <>
__device__ __forceinline__ __half Int4FromFloat<__half>(float x) {
  return __float2half(x);
}
template <>
__device__ __forceinline__ __nv_bfloat16 Int4FromFloat<__nv_bfloat16>(
    float x) {
  return __float2bfloat16(x);
}

// One block per (row, group). Block-reduces the amax over the group's elements
// and writes the fp16 scale. A block per group rather than a grid-stride
// reduction because there are n*num_groups of them and each is small (group is
// typically 128) -- one block owns one group's reduction end to end.
template <typename H>
__global__ void Int4GroupScaleKernel(const H* __restrict__ w, int64_t n,
                                     int64_t k, int group, int groups_per_row,
                                     H* __restrict__ scales) {
  const int row = blockIdx.x / groups_per_row;
  const int g = blockIdx.x % groups_per_row;
  if (row >= n) return;

  const int64_t base = static_cast<int64_t>(row) * k +
                       static_cast<int64_t>(g) * group;

  float local = 0.0f;
  for (int i = threadIdx.x; i < group; i += blockDim.x) {
    local = fmaxf(local, fabsf(Widen(w[base + i])));
  }

  __shared__ float tile[256];
  tile[threadIdx.x] = local;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      tile[threadIdx.x] = fmaxf(tile[threadIdx.x], tile[threadIdx.x + stride]);
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    const float amax = tile[0];
    // An all-zero group gets scale 1 so the dequant multiplies by something;
    // its outputs are zero regardless, so the value does not matter beyond
    // avoiding a divide-by-zero in the quantize kernel.
    const float scale = amax > 0.0f ? amax / kInt4SymmetricMax : 1.0f;
    scales[static_cast<int64_t>(row) * groups_per_row + g] =
        Int4FromFloat<H>(scale);
  }
}

// Quantizes and packs. Each thread owns one byte = two consecutive elements
// (indices 2*b and 2*b+1), so the two nibbles of a byte never race. k is even
// (checked by the caller), so a pair never straddles a row boundary and the
// two elements always share a group.
template <typename H>
__global__ void QuantizeInt4Kernel(const H* __restrict__ w, int64_t n,
                                   int64_t k, int group, int groups_per_row,
                                   const H* __restrict__ scales,
                                   uint8_t* __restrict__ q) {
  const int64_t nbytes = n * (k / 2);
  for (int64_t b = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       b < nbytes;
       b += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const int64_t e0 = 2 * b;
    const int64_t row = e0 / k;
    const int col0 = static_cast<int>(e0 % k);

    const float s0 =
        Widen(scales[static_cast<int64_t>(row) * groups_per_row +
                     (col0 / group)]);
    const float s1 =
        Widen(scales[static_cast<int64_t>(row) * groups_per_row +
                     ((col0 + 1) / group)]);

    // Round to nearest even, then clamp to the symmetric range. __float2int_rn
    // rounds ties to even, which is what keeps a uniform tensor quantizing
    // symmetrically rather than biasing up.
    int q0 = __float2int_rn(Widen(w[e0]) / s0);
    int q1 = __float2int_rn(Widen(w[e0 + 1]) / s1);
    q0 = q0 < -7 ? -7 : (q0 > 7 ? 7 : q0);
    q1 = q1 < -7 ? -7 : (q1 > 7 ? 7 : q1);

    // Two's-complement 4-bit: for v in [-8,7], (v & 0xF) is exactly the nibble.
    q[b] = static_cast<uint8_t>((q0 & 0xF) | ((q1 & 0xF) << 4));
  }
}

// The inverse: one byte in, two fp16 out, each scaled by its own group.
template <typename H>
__global__ void DequantizeInt4Kernel(
    const uint8_t* __restrict__ q, int64_t n, int64_t k, int group,
    int groups_per_row, const H* __restrict__ scales, H* __restrict__ dst) {
  const int64_t nbytes = n * (k / 2);
  for (int64_t b = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       b < nbytes;
       b += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const uint8_t byte = q[b];
    int q0 = byte & 0xF;       // sign-extend the nibble:
    if (q0 >= 8) q0 -= 16;     // 0x8..0xF map to -8..-1.
    int q1 = byte >> 4;
    if (q1 >= 8) q1 -= 16;

    const int64_t e0 = 2 * b;
    const int64_t row = e0 / k;
    const int col0 = static_cast<int>(e0 % k);

    const float s0 =
        Widen(scales[static_cast<int64_t>(row) * groups_per_row +
                     (col0 / group)]);
    const float s1 =
        Widen(scales[static_cast<int64_t>(row) * groups_per_row +
                     ((col0 + 1) / group)]);

    dst[e0] = Int4FromFloat<H>(q0 * s0);
    dst[e0 + 1] = Int4FromFloat<H>(q1 * s1);
  }
}

}  // namespace

Status QuantizeF16ToInt4(const TensorView& src, const TensorView& dst,
                         const TensorView& scales, Stream stream) {
  INFERX_RETURN_IF_ERROR(CheckDeviceTensor(src, DataType::kFloat16, "src"));
  INFERX_RETURN_IF_ERROR(CheckDeviceTensor(dst, DataType::kInt4, "dst"));
  INFERX_RETURN_IF_ERROR(
      CheckDeviceTensor(scales, DataType::kFloat16, "scales"));

  if (src.Rank() != 2 || dst.Rank() != 2 || scales.Rank() != 2) {
    return InvalidArgumentError(
        "QuantizeF16ToInt4: src/dst/scales must be rank 2");
  }

  const int64_t n = src.Dim(0);
  const int64_t k = src.Dim(1);

  if (dst.Dim(0) != n || dst.Dim(1) != k) {
    return InvalidArgumentError("QuantizeF16ToInt4: dst is [", dst.Dim(0), ", ",
                                dst.Dim(1), "], expected [", n, ", ", k, "]");
  }
  if (scales.Dim(0) != n) {
    return InvalidArgumentError("QuantizeF16ToInt4: scales has ",
                                scales.Dim(0), " rows, expected ", n);
  }
  if (k % 2 != 0) {
    return InvalidArgumentError(
        "QuantizeF16ToInt4: k must be even (two int4 pack into one byte), got ",
        k);
  }

  const int groups_per_row = static_cast<int>(scales.Dim(1));
  if (groups_per_row <= 0 || k % groups_per_row != 0) {
    return InvalidArgumentError(
        "QuantizeF16ToInt4: scales has ", groups_per_row,
        " groups per row, which must divide k=", k);
  }
  const int group = static_cast<int>(k / groups_per_row);

  if (n == 0 || k == 0) return OkStatus();

  // One block per group for the reduction. blockDim covers a group with
  // stride; group=128 lands one thread per element, larger groups stride. The
  // halving reduction inside the kernel only combines every lane when blockDim
  // is a power of two, so round the group up to the next power of two (floor 32,
  // ceiling 256 to fit the shared-memory tile); threads past the group load
  // nothing and contribute 0 to the max.
  int block = 32;
  while (block < group && block < 256) block <<= 1;
  Int4GroupScaleKernel<__half><<<static_cast<int>(n) * groups_per_row, block, 0,
                         stream>>>(static_cast<const __half*>(src.Data()), n, k,
                                   group, groups_per_row,
                                   static_cast<__half*>(scales.Data()));
  INFERX_CUDA_RETURN_IF_ERROR(cudaGetLastError());

  QuantizeInt4Kernel<__half><<<GridFor(n * (k / 2)), kBlock, 0, stream>>>(
      static_cast<const __half*>(src.Data()), n, k, group, groups_per_row,
      static_cast<const __half*>(scales.Data()),
      static_cast<uint8_t*>(dst.Data()));
  INFERX_CUDA_RETURN_IF_ERROR(cudaGetLastError());

  return OkStatus();
}

Status DequantizeInt4ToF16(const TensorView& src, const TensorView& scales,
                           const TensorView& dst, Stream stream) {
  INFERX_RETURN_IF_ERROR(CheckDeviceTensor(src, DataType::kInt4, "src"));
  INFERX_RETURN_IF_ERROR(
      CheckDeviceTensor(scales, DataType::kFloat16, "scales"));
  INFERX_RETURN_IF_ERROR(CheckDeviceTensor(dst, DataType::kFloat16, "dst"));

  if (src.Rank() != 2 || dst.Rank() != 2 || scales.Rank() != 2) {
    return InvalidArgumentError(
        "DequantizeInt4ToF16: src/dst/scales must be rank 2");
  }

  const int64_t n = src.Dim(0);
  const int64_t k = src.Dim(1);

  if (dst.Dim(0) != n || dst.Dim(1) != k) {
    return InvalidArgumentError("DequantizeInt4ToF16: dst is [", dst.Dim(0),
                                ", ", dst.Dim(1), "], expected [", n, ", ", k,
                                "]");
  }
  if (scales.Dim(0) != n) {
    return InvalidArgumentError("DequantizeInt4ToF16: scales has ",
                                scales.Dim(0), " rows, expected ", n);
  }
  if (k % 2 != 0) {
    return InvalidArgumentError(
        "DequantizeInt4ToF16: k must be even, got ", k);
  }

  const int groups_per_row = static_cast<int>(scales.Dim(1));
  if (groups_per_row <= 0 || k % groups_per_row != 0) {
    return InvalidArgumentError(
        "DequantizeInt4ToF16: scales has ", groups_per_row,
        " groups per row, which must divide k=", k);
  }
  const int group = static_cast<int>(k / groups_per_row);

  if (n == 0 || k == 0) return OkStatus();

  DequantizeInt4Kernel<__half><<<GridFor(n * (k / 2)), kBlock, 0, stream>>>(
      static_cast<const uint8_t*>(src.Data()), n, k, group, groups_per_row,
      static_cast<const __half*>(scales.Data()),
      static_cast<__half*>(dst.Data()));
  INFERX_CUDA_RETURN_IF_ERROR(cudaGetLastError());

  return OkStatus();
}

Status QuantizeBf16ToInt4(const TensorView& src, const TensorView& dst,
                          const TensorView& scales, Stream stream) {
  // The bf16 mirror of QuantizeF16ToInt4, so W4A16 stays in the model's native
  // dtype instead of casting bf16 <-> f16 around every linear. The block-size
  // rounding note in QuantizeF16ToInt4 applies here too.
  INFERX_RETURN_IF_ERROR(CheckDeviceTensor(src, DataType::kBFloat16, "src"));
  INFERX_RETURN_IF_ERROR(CheckDeviceTensor(dst, DataType::kInt4, "dst"));
  INFERX_RETURN_IF_ERROR(
      CheckDeviceTensor(scales, DataType::kBFloat16, "scales"));

  if (src.Rank() != 2 || dst.Rank() != 2 || scales.Rank() != 2) {
    return InvalidArgumentError(
        "QuantizeBf16ToInt4: src/dst/scales must be rank 2");
  }
  const int64_t n = src.Dim(0);
  const int64_t k = src.Dim(1);
  if (dst.Dim(0) != n || dst.Dim(1) != k) {
    return InvalidArgumentError("QuantizeBf16ToInt4: dst is [", dst.Dim(0), ", ",
                                dst.Dim(1), "], expected [", n, ", ", k, "]");
  }
  if (scales.Dim(0) != n) {
    return InvalidArgumentError("QuantizeBf16ToInt4: scales has ",
                                scales.Dim(0), " rows, expected ", n);
  }
  if (k % 2 != 0) {
    return InvalidArgumentError("QuantizeBf16ToInt4: k must be even, got ", k);
  }
  const int groups_per_row = static_cast<int>(scales.Dim(1));
  if (groups_per_row <= 0 || k % groups_per_row != 0) {
    return InvalidArgumentError(
        "QuantizeBf16ToInt4: scales has ", groups_per_row,
        " groups per row, which must divide k=", k);
  }
  const int group = static_cast<int>(k / groups_per_row);
  if (n == 0 || k == 0) return OkStatus();

  int block = 32;
  while (block < group && block < 256) block <<= 1;
  Int4GroupScaleKernel<__nv_bfloat16>
      <<<static_cast<int>(n) * groups_per_row, block, 0, stream>>>(
          static_cast<const __nv_bfloat16*>(src.Data()), n, k, group,
          groups_per_row,
          static_cast<__nv_bfloat16*>(scales.Data()));
  INFERX_CUDA_RETURN_IF_ERROR(cudaGetLastError());

  QuantizeInt4Kernel<__nv_bfloat16>
      <<<GridFor(n * (k / 2)), kBlock, 0, stream>>>(
          static_cast<const __nv_bfloat16*>(src.Data()), n, k, group,
          groups_per_row,
          static_cast<const __nv_bfloat16*>(scales.Data()),
          static_cast<uint8_t*>(dst.Data()));
  INFERX_CUDA_RETURN_IF_ERROR(cudaGetLastError());
  return OkStatus();
}

Status DequantizeInt4ToBf16(const TensorView& src, const TensorView& scales,
                            const TensorView& dst, Stream stream) {
  INFERX_RETURN_IF_ERROR(CheckDeviceTensor(src, DataType::kInt4, "src"));
  INFERX_RETURN_IF_ERROR(
      CheckDeviceTensor(scales, DataType::kBFloat16, "scales"));
  INFERX_RETURN_IF_ERROR(CheckDeviceTensor(dst, DataType::kBFloat16, "dst"));

  if (src.Rank() != 2 || dst.Rank() != 2 || scales.Rank() != 2) {
    return InvalidArgumentError(
        "DequantizeInt4ToBf16: src/dst/scales must be rank 2");
  }
  const int64_t n = src.Dim(0);
  const int64_t k = src.Dim(1);
  if (dst.Dim(0) != n || dst.Dim(1) != k) {
    return InvalidArgumentError("DequantizeInt4ToBf16: dst is [", dst.Dim(0),
                                ", ", dst.Dim(1), "], expected [", n, ", ", k,
                                "]");
  }
  if (scales.Dim(0) != n) {
    return InvalidArgumentError("DequantizeInt4ToBf16: scales has ",
                                scales.Dim(0), " rows, expected ", n);
  }
  if (k % 2 != 0) {
    return InvalidArgumentError("DequantizeInt4ToBf16: k must be even, got ", k);
  }
  const int groups_per_row = static_cast<int>(scales.Dim(1));
  if (groups_per_row <= 0 || k % groups_per_row != 0) {
    return InvalidArgumentError(
        "DequantizeInt4ToBf16: scales has ", groups_per_row,
        " groups per row, which must divide k=", k);
  }
  const int group = static_cast<int>(k / groups_per_row);
  if (n == 0 || k == 0) return OkStatus();

  DequantizeInt4Kernel<__nv_bfloat16>
      <<<GridFor(n * (k / 2)), kBlock, 0, stream>>>(
          static_cast<const uint8_t*>(src.Data()), n, k, group, groups_per_row,
          static_cast<const __nv_bfloat16*>(scales.Data()),
          static_cast<__nv_bfloat16*>(dst.Data()));
  INFERX_CUDA_RETURN_IF_ERROR(cudaGetLastError());
  return OkStatus();
}

}  // namespace inferx::kernels
