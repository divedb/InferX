#ifndef INFERX_OPS_CUDA_RMS_NORM_H_
#define INFERX_OPS_CUDA_RMS_NORM_H_

#include "inferx/core/status.h"
#include "inferx/core/tensor.h"
#include "inferx/ops/execution_context.h"
#include "inferx/ops/rms_norm.h"

namespace inferx::ops::cuda {

/// \brief FlashInfer-backed RMSNorm for CUDA devices.
///
/// Serves the fp32/fp16/bf16 activations FlashInfer's norm kernels serve,
/// dispatching between norm::RMSNorm and norm::GemmaRMSNorm for the
/// 1 + weight variant. Kernel selection and dtype support stay entirely
/// inside this backend.
Status RmsNorm(ExecutionContext& ctx, const Tensor& x, const Tensor& weight, Tensor& out,
               const RMSNormConfig& config);

}  // namespace inferx::ops::cuda

#endif  // INFERX_OPS_CUDA_RMS_NORM_H_
