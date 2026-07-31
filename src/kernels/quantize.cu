#include "inferx/kernels/quantize.h"

#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include "inferx/core/cuda_utils.h"

namespace inferx::kernels {
namespace {

constexpr int kBlock = 256;

// Grid-stride amax with a block reduction, finished with one atomic per block.
//
// atomicMax on the *bit pattern* rather than on the float: IEEE-754 orders
// non-negative floats identically to their unsigned bit patterns, and we only
// ever feed this fabsf() output, so the integer maximum is the float maximum.
// CUDA has no atomicMax for float, and doing this with a CAS loop would be
// slower for no gain.
__global__ void AbsMaxKernel(const __half* __restrict__ src, int64_t n,
                             float* __restrict__ out) {
  __shared__ float tile[kBlock];

  float local = 0.0f;

  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < n; i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    local = fmaxf(local, fabsf(__half2float(src[i])));
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

__global__ void QuantizeKernel(const __half* __restrict__ src,
                               __nv_fp8_storage_t* __restrict__ dst, int64_t n,
                               const float* __restrict__ scale) {
  // Read once per thread rather than per element: it is the same value for the
  // whole launch and the compiler cannot know that through a pointer.
  const float inv = 1.0f / *scale;

  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < n; i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    // __NV_SATFINITE clamps to ±448 instead of producing inf/NaN, so a scale
    // that turns out too small costs accuracy rather than correctness.
    dst[i] = __nv_cvt_float_to_fp8(__half2float(src[i]) * inv, __NV_SATFINITE,
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

}  // namespace

Status ComputeF8Scale(const TensorView& src, float* scale_dev,
                      cudaStream_t stream) {
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
                           const float* scale_dev, cudaStream_t stream) {
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

}  // namespace inferx::kernels
