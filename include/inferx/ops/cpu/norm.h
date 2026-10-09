#ifndef INFERX_OPS_CPU_NORM_H_
#define INFERX_OPS_CPU_NORM_H_

#include "inferx/core/status.h"
#include "inferx/core/tensor.h"
#include "inferx/ops/norm.h"
#include "inferx/ops/op_context.h"

namespace inferx::ops::cpu {

/// \brief Highway-vectorized RMSNorm for the host.
///
/// Serves fp32/fp16/bf16, computing every row in fp32 and converting at the
/// boundaries for the half-precision dtypes. The SIMD width, multi-target
/// dispatch, and conversions stay entirely inside this backend.
Status RmsNorm(OpContext& ctx, const Tensor& x, const Tensor& weight, Tensor& out,
               const NormConfig& config);

/// \brief Fused add + RMSNorm for the host: the residual update rounds to
///        the activation dtype first and the normalization reads the rounded
///        sum, exactly like the CUDA kernel.
Status AddRmsNorm(OpContext& ctx, const Tensor& x, Tensor& residual, const Tensor& weight,
                  Tensor& out, const NormConfig& config);

/// \brief Scalar two-pass LayerNorm for the host, fp32 accumulation with
///        fp16/bf16 converted row-wise like the RMS path.
Status LayerNorm(OpContext& ctx, const Tensor& x, const Tensor& weight,
                 const std::optional<Tensor>& bias, Tensor& out, const NormConfig& config);

}  // namespace inferx::ops::cpu

#endif  // INFERX_OPS_CPU_NORM_H_
