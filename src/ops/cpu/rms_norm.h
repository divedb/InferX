#ifndef INFERX_OPS_CPU_RMS_NORM_H_
#define INFERX_OPS_CPU_RMS_NORM_H_

#include "inferx/core/status.h"
#include "inferx/core/tensor.h"
#include "inferx/ops/execution_context.h"
#include "inferx/ops/rms_norm.h"

namespace inferx::ops::cpu {

/// \brief Highway-vectorized RMSNorm for the host.
///
/// Serves fp32/fp16/bf16, computing every row in fp32 and converting at the
/// boundaries for the half-precision dtypes. The SIMD width, multi-target
/// dispatch, and conversions stay entirely inside this backend.
Status RmsNorm(ExecutionContext& ctx, const Tensor& x, const Tensor& weight, Tensor& out,
               const RMSNormConfig& config);

}  // namespace inferx::ops::cpu

#endif  // INFERX_OPS_CPU_RMS_NORM_H_
