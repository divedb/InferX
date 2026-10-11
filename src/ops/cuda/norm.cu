#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <optional>
#include <type_traits>

#include "dtype_dispatch.cuh"
#include "flashinfer/norm.cuh"
#include "inferx/ops/cuda/status.h"
#include "inferx/ops/cuda/norm.h"

namespace inferx::ops::cuda {
namespace {

/// Block-wide sum over one row; result valid in every thread afterwards.
__device__ float BlockReduceSumRow(float value) {
  __shared__ float warp_sums[32];
  const int lane = threadIdx.x;
  const int warp = threadIdx.x / 32;
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(0xffffffffu, value, offset);
  }
  if (lane == 0) warp_sums[warp] = value;
  __syncthreads();
  if (warp == 0) {
    value = lane < (blockDim.x + 31) / 32 ? warp_sums[lane] : 0.0f;
    for (int offset = 16; offset > 0; offset >>= 1) {
      value += __shfl_down_sync(0xffffffffu, value, offset);
    }
    if (lane == 0) warp_sums[0] = value;
  }
  __syncthreads();
  return warp_sums[0];
}

/// Llama/Qwen reference rounding: normalize in fp32, cast back to the
/// activation dtype, then multiply by the learned scale (each product
/// rounds). With AddResidual the dtype-rounded sum is stored into
/// `residual` and the normalization reads that rounded sum.
template <typename T, bool AddResidual = false>
__global__ void RoundedRmsNorm(const T* input, const T* weight, T* output, int64_t dim,
                               float eps, float weight_bias, T* residual = nullptr) {
  __shared__ float inv_rms;
  const int64_t base = static_cast<int64_t>(blockIdx.x) * dim;
  float sum = 0.0f;
  for (int64_t j = threadIdx.x; j < dim; j += blockDim.x) {
    T value = input[base + j];
    if constexpr (AddResidual) {
      value = static_cast<T>(static_cast<float>(value) +
                             static_cast<float>(residual[base + j]));
      residual[base + j] = value;
    }
    const float x = static_cast<float>(value);
    sum += x * x;
  }
  sum = BlockReduceSumRow(sum);
  if (threadIdx.x == 0) inv_rms = rsqrtf(sum / static_cast<float>(dim) + eps);
  __syncthreads();
  for (int64_t j = threadIdx.x; j < dim; j += blockDim.x) {
    const T value = AddResidual ? residual[base + j] : input[base + j];
    const T normalized = static_cast<T>(static_cast<float>(value) * inv_rms);
    output[base + j] = static_cast<T>(static_cast<float>(normalized) *
                                      (static_cast<float>(weight[j]) + weight_bias));
  }
}

/// Launches the rounded kernel for the context dtype; bf16/f16 round for
/// real, fp32 casts are identities.
template <typename T, bool AddResidual = false>
Status LaunchRounded(OpContext& ctx, const Tensor& input, const Tensor& weight,
                     Tensor& output, const NormConfig& config, Tensor* residual = nullptr) {
  const int64_t rows = input.Numel() / input.Dim(input.Rank() - 1);
  RoundedRmsNorm<T, AddResidual><<<static_cast<uint32_t>(rows), 256, 0, ctx.GetStream().As<cudaStream_t>()>>>(
      static_cast<const T*>(input.Data()), static_cast<const T*>(weight.Data()),
      static_cast<T*>(output.Data()), input.Dim(input.Rank() - 1), config.eps,
      config.plus_one ? 1.0f : 0.0f,
      residual != nullptr ? static_cast<T*>(residual->Data()) : nullptr);
  return CudaStatus(cudaGetLastError(), "rounded rms norm");
}

}  // namespace

Status RmsNorm(OpContext& ctx, const Tensor& input, const Tensor& weight, Tensor& output,
               const NormConfig& config) {
  INFERX_RETURN_IF_ERROR(ctx.Runtime().Activate());
  if (config.round_before_weight) {
    Status launched = DispatchDtype(
        input.GetDataType(),
        [](auto* tag, OpContext& ctx, const Tensor& input, const Tensor& weight,
           Tensor& output, const NormConfig& config) {
          using T = std::remove_pointer_t<decltype(tag)>;
          return LaunchRounded<T>(ctx, input, weight, output, config);
        },
        ctx, input, weight, output, config);
    if (launched.ok()) return launched;
    // Dtype the rounded kernel cannot serve falls through to the fused op.
  }
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
                                                ctx.GetStream().As<cudaStream_t>())
            : flashinfer::norm::RMSNorm<T>(in, w, out, n_rows, hidden, hidden, hidden,
                                           config.eps, /*enable_pdl=*/false, ctx.GetStream().As<cudaStream_t>());
    return CudaStatus(err, config.plus_one ? "FlashInfer GemmaRMSNorm" : "FlashInfer RMSNorm");
  }, ctx, input, weight, output, config);
}

Status AddRmsNorm(OpContext& ctx, const Tensor& input, Tensor& residual, const Tensor& weight,
                  Tensor& output, const NormConfig& config) {
  INFERX_RETURN_IF_ERROR(ctx.Runtime().Activate());
  if (config.round_before_weight &&
      (input.GetDataType() == DataType::kBFloat16 ||
       input.GetDataType() == DataType::kFloat16)) {
    // Opt-in reference rounding: the dtype-rounded sum lands in the
    // residual, the normalization reads that rounded sum, and the scale
    // multiply rounds again.
    Status launched = DispatchDtype(
        input.GetDataType(),
        [](auto* tag, OpContext& ctx, const Tensor& input, Tensor& residual,
           const Tensor& weight, Tensor& output, const NormConfig& config) {
          using T = std::remove_pointer_t<decltype(tag)>;
          return LaunchRounded<T, /*AddResidual=*/true>(ctx, input, weight, output,
                                                        config, &residual);
        },
        ctx, input, residual, weight, output, config);
    if (launched.ok()) return launched;
  }
  return DispatchDtype(input.GetDataType(), [](auto* tag, OpContext& ctx,
      const Tensor& input, Tensor& residual, const Tensor& weight, Tensor& output,
      const NormConfig& config) {
    using T = std::remove_pointer_t<decltype(tag)>;
    // FlashInfer overwrites its input with the normalized result. Use output
    // as that buffer, preserving input unless the caller explicitly aliases it.
    if (input.Data() != output.Data()) {
      INFERX_RETURN_IF_ERROR(CudaStatus(
          cudaMemcpyAsync(output.Data(), input.Data(), input.Numel() * sizeof(T),
                          cudaMemcpyDeviceToDevice, ctx.GetStream().As<cudaStream_t>()),
          "AddRmsNorm input copy"));
    }
    T* out = static_cast<T*>(output.Data());
    T* r = static_cast<T*>(residual.Data());
    T* w = const_cast<T*>(static_cast<const T*>(weight.Data()));
    const uint32_t hidden = static_cast<uint32_t>(input.Dim(input.Rank() - 1));
    const uint32_t rows = static_cast<uint32_t>(input.Numel() / hidden);
    const cudaError_t err = config.plus_one
        ? flashinfer::norm::GemmaFusedAddRMSNorm<T>(
              out, r, w, rows, hidden, hidden, hidden, config.eps, false, ctx.GetStream().As<cudaStream_t>())
        : flashinfer::norm::FusedAddRMSNorm<T>(
              out, r, w, rows, hidden, hidden, hidden, config.eps, false, ctx.GetStream().As<cudaStream_t>());
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
        static_cast<T*>(output.Data()), n_rows, hidden, config.eps, ctx.GetStream().As<cudaStream_t>());
    return CudaStatus(err, "FlashInfer LayerNorm");
  }, ctx, input, weight, bias, output, config);
}

}  // namespace inferx::ops::cuda
