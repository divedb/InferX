#ifndef INFERX_OPS_CUDA_TYPE_CONVERSION_CUH_
#define INFERX_OPS_CUDA_TYPE_CONVERSION_CUH_

#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace inferx::ops::cuda {

// Scalar conversions used by CUDA kernels; narrowing uses round-to-nearest-even.
__device__ __forceinline__ float ToFloat(float v) { return v; }
__device__ __forceinline__ float ToFloat(__half v) { return __half2float(v); }
__device__ __forceinline__ float ToFloat(__nv_bfloat16 v) { return __bfloat162float(v); }

template <typename T>
__device__ __forceinline__ T FromFloat(float v);
template <>
__device__ __forceinline__ float FromFloat<float>(float v) {
  return v;
}
template <>
__device__ __forceinline__ __half FromFloat<__half>(float v) {
  return __float2half_rn(v);
}
template <>
__device__ __forceinline__ __nv_bfloat16 FromFloat<__nv_bfloat16>(float v) {
  return __float2bfloat16_rn(v);
}

}  // namespace inferx::ops::cuda

#endif  // INFERX_OPS_CUDA_TYPE_CONVERSION_CUH_
