#include <cuda_runtime.h>

#include <cstdint>
#include <optional>
#include <type_traits>

#include "dtype_dispatch.cuh"
#include "flashinfer/norm.cuh"
#include "inferx/ops/cuda/status.h"
#include "inferx/ops/cuda/norm.h"

namespace inferx::ops::cuda {
Status RmsNorm(OpContext& ctx, const Tensor& input, const Tensor& weight, Tensor& output,
               const NormConfig& config) {
  INFERX_RETURN_IF_ERROR(ctx.Runtime().Activate());
  return DispatchDtype(input.GetDataType(), [](auto* tag, OpContext& ctx,
      const Tensor& input, const Tensor& weight, Tensor& output, const NormConfig& config) {
    using T = std::remove_pointer_t<decltype(tag)>;
    // FlashInfer's launchers take non-const pointers although the kernels
    // never write the input or the weight. GemmaRMSNorm is the same kernel
    // launched with weight_bias = 1, i.e. the (1 + w) scale.
    T* in = const_cast<T*>(static_cast<const T*>(input.Data()));
    T* w = const_cast<T*>(static_cast<const T*>(weight.Data()));
    T* out = static_cast<T*>(output.Data());
    const uint32_t hidden = static_cast<uint32_t>(input.Dim(input.Rank() - 1));
    const uint32_t n_rows = static_cast<uint32_t>(input.Numel() / hidden);
    const cudaError_t err =
        config.plus_one
            ? flashinfer::norm::GemmaRMSNorm<T>(in, w, out, n_rows, hidden, hidden, hidden,
                                                config.eps, /*enable_pdl=*/false,
                                                ctx.GetStream())
            : flashinfer::norm::RMSNorm<T>(in, w, out, n_rows, hidden, hidden, hidden,
                                           config.eps, /*enable_pdl=*/false, ctx.GetStream());
    return CudaStatus(err, config.plus_one ? "FlashInfer GemmaRMSNorm" : "FlashInfer RMSNorm");
  }, ctx, input, weight, output, config);
}

Status AddRmsNorm(OpContext& ctx, const Tensor& input, Tensor& residual, const Tensor& weight,
                  Tensor& output, const NormConfig& config) {
  INFERX_RETURN_IF_ERROR(ctx.Runtime().Activate());
  return DispatchDtype(input.GetDataType(), [](auto* tag, OpContext& ctx,
      const Tensor& input, Tensor& residual, const Tensor& weight, Tensor& output,
      const NormConfig& config) {
    using T = std::remove_pointer_t<decltype(tag)>;
    // FlashInfer overwrites its input with the normalized result. Use output
    // as that buffer, preserving input unless the caller explicitly aliases it.
    if (input.Data() != output.Data()) {
      INFERX_RETURN_IF_ERROR(CudaStatus(
          cudaMemcpyAsync(output.Data(), input.Data(), input.Numel() * sizeof(T),
                          cudaMemcpyDeviceToDevice, ctx.GetStream()),
          "AddRmsNorm input copy"));
    }
    T* out = static_cast<T*>(output.Data());
    T* r = static_cast<T*>(residual.Data());
    T* w = const_cast<T*>(static_cast<const T*>(weight.Data()));
    const uint32_t hidden = static_cast<uint32_t>(input.Dim(input.Rank() - 1));
    const uint32_t rows = static_cast<uint32_t>(input.Numel() / hidden);
    const cudaError_t err = config.plus_one
        ? flashinfer::norm::GemmaFusedAddRMSNorm<T>(
              out, r, w, rows, hidden, hidden, hidden, config.eps, false, ctx.GetStream())
        : flashinfer::norm::FusedAddRMSNorm<T>(
              out, r, w, rows, hidden, hidden, hidden, config.eps, false, ctx.GetStream());
    return CudaStatus(err, config.plus_one ? "FlashInfer GemmaFusedAddRMSNorm"
                                          : "FlashInfer FusedAddRMSNorm");
  }, ctx, input, residual, weight, output, config);
}

Status LayerNorm(OpContext& ctx, const Tensor& input, const Tensor& weight,
                 const std::optional<Tensor>& bias, Tensor& output, const NormConfig& config) {
  INFERX_RETURN_IF_ERROR(ctx.Runtime().Activate());
  return DispatchDtype(input.GetDataType(), [](auto* tag, OpContext& ctx,
      const Tensor& input, const Tensor& weight, const std::optional<Tensor>& bias,
      Tensor& output, const NormConfig& config) {
    using T = std::remove_pointer_t<decltype(tag)>;
    const uint32_t hidden = static_cast<uint32_t>(input.Dim(input.Rank() - 1));
    const uint32_t n_rows = static_cast<uint32_t>(input.Numel() / hidden);
    const cudaError_t err = flashinfer::norm::LayerNorm<T, T>(
        const_cast<T*>(static_cast<const T*>(input.Data())),
        const_cast<T*>(static_cast<const T*>(weight.Data())),
        bias ? const_cast<T*>(static_cast<const T*>(bias->Data())) : nullptr,
        static_cast<T*>(output.Data()), n_rows, hidden, config.eps, ctx.GetStream());
    return CudaStatus(err, "FlashInfer LayerNorm");
  }, ctx, input, weight, bias, output, config);
}

}  // namespace inferx::ops::cuda
