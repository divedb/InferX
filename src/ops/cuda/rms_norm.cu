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
  const cudaError_t err =
      config.plus_one_weight
          ? flashinfer::norm::GemmaRMSNorm<T>(input, scale, output, rows, dim, dim, dim,
                                              config.eps, /*enable_pdl=*/false, ctx.stream())
          : flashinfer::norm::RMSNorm<T>(input, scale, output, rows, dim, dim, dim, config.eps,
                                         /*enable_pdl=*/false, ctx.stream());
  return CudaError(err, config.plus_one_weight ? "gemma rms norm" : "rms norm");
}

}  // namespace

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
