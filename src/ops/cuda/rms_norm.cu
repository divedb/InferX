#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "flashinfer/norm.cuh"
#include "inferx/core/dtype.h"
#include "ops/cuda/rms_norm.h"

namespace inferx::ops::cuda {
namespace {

Status CudaError(cudaError_t err, const char* what) {
  if (err != cudaSuccess) {
    return InternalError(what, " failed: ", cudaGetErrorString(err));
  }
  return OkStatus();
}

// Llama/Qwen normalize in float, cast back to the activation dtype, then
// multiply by the learned scale. FlashInfer's RMSNorm fuses away that cast.
template <typename T, bool AddResidual = false>
__global__ void RoundedRmsNorm(const T* input, const T* weight, T* output,
                               int64_t dim, float eps, float bias, T* residual = nullptr) {
  __shared__ float inv_rms;
  const int64_t base = static_cast<int64_t>(blockIdx.x) * dim;
  float sum = 0.0f;
  for (int64_t j = threadIdx.x; j < dim; j += blockDim.x) {
    T value = input[base + j];
    if constexpr (AddResidual) {
      value = static_cast<T>(static_cast<float>(value) + static_cast<float>(residual[base + j]));
      residual[base + j] = value;
    }
    const float x = static_cast<float>(value);
    sum += x * x;
  }
  sum = tensorrt_llm::common::blockReduceSum(sum);
  if (threadIdx.x == 0) inv_rms = rsqrtf(sum / static_cast<float>(dim) + eps);
  __syncthreads();
  for (int64_t j = threadIdx.x; j < dim; j += blockDim.x) {
    const T value = AddResidual ? residual[base + j] : input[base + j];
    const T normalized = static_cast<T>(static_cast<float>(value) * inv_rms);
    output[base + j] = static_cast<T>(static_cast<float>(normalized) *
                                     (static_cast<float>(weight[j]) + bias));
  }
}

/// \brief Launches FlashInfer's norm kernel for one element type.
///
/// Tensors are contiguous, so both strides are the hidden size. FlashInfer's
/// launcher takes a non-const weight pointer although the kernel never writes
/// it, and supports `x` aliasing `out` for in-place updates.
template <typename T>
Status RmsNormImpl(ExecutionContext& ctx, const Tensor& x, const Tensor& weight, Tensor& out,
                   const RMSNormConfig& config) {
  INFERX_RETURN_IF_ERROR(ctx.runtime().Activate());
  const uint32_t rows = static_cast<uint32_t>(x.Dim(0));
  const uint32_t dim = static_cast<uint32_t>(x.Dim(1));
  // FlashInfer's launchers take non-const pointers although the kernel never
  // writes the input or the weight.
  T* input = const_cast<T*>(static_cast<const T*>(x.Data()));
  T* scale = const_cast<T*>(static_cast<const T*>(weight.Data()));
  T* output = static_cast<T*>(out.Data());
  if (config.round_before_weight) {
    RoundedRmsNorm<<<rows, 256, 0, ctx.stream()>>>(
        input, scale, output, dim, config.eps, config.plus_one_weight ? 1.0f : 0.0f);
    return CudaError(cudaGetLastError(), "rounded rms norm");
  }
  const cudaError_t err =
      config.plus_one_weight
          ? flashinfer::norm::GemmaRMSNorm<T>(input, scale, output, rows, dim, dim, dim,
                                              config.eps, /*enable_pdl=*/false, ctx.stream())
          : flashinfer::norm::RMSNorm<T>(input, scale, output, rows, dim, dim, dim, config.eps,
                                         /*enable_pdl=*/false, ctx.stream());
  return CudaError(err, config.plus_one_weight ? "gemma rms norm" : "rms norm");
}

}  // namespace

Status AddRmsNorm(ExecutionContext& ctx, const Tensor& x, Tensor& residual,
                  const Tensor& weight, Tensor& out, const RMSNormConfig& config) {
  INFERX_RETURN_IF_ERROR(ctx.runtime().Activate());
  RoundedRmsNorm<__nv_bfloat16, true><<<x.Dim(0), 256, 0, ctx.stream()>>>(
      static_cast<const __nv_bfloat16*>(x.Data()),
      static_cast<const __nv_bfloat16*>(weight.Data()),
      static_cast<__nv_bfloat16*>(out.Data()), x.Dim(1), config.eps,
      config.plus_one_weight ? 1.0f : 0.0f, static_cast<__nv_bfloat16*>(residual.Data()));
  return CudaError(cudaGetLastError(), "add rms norm");
}

Status RmsNorm(ExecutionContext& ctx, const Tensor& x, const Tensor& weight, Tensor& out,
               const RMSNormConfig& config) {
  switch (x.GetDataType()) {
    case DataType::kFloat:
      return RmsNormImpl<float>(ctx, x, weight, out, config);
    case DataType::kFloat16:
      return RmsNormImpl<__half>(ctx, x, weight, out, config);
    case DataType::kBFloat16:
      return RmsNormImpl<__nv_bfloat16>(ctx, x, weight, out, config);
    default:
      return UnimplementedError("RmsNorm on CUDA does not support dtype ",
                                DataTypeName(x.GetDataType()));
  }
}

}  // namespace inferx::ops::cuda
