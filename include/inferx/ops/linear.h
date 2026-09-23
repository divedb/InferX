#pragma once

#include "inferx/core/status.h"
#include "inferx/core/tensor.h"
#include "inferx/ops/execution_context.h"

namespace inferx::ops {

/// \brief Bias-free projection: out[t, :] <- x[t, :] @ weight^T.
///
/// `weight` is stored exactly as checkpoints store it, [out_features,
/// in_features], so no transposition happens at load time. On CUDA this is a
/// vendor GEMM with fp32 accumulation; `out` must not alias `x` or `weight`.
///
/// \param ctx     Execution context; all tensors must live on ctx.device().
/// \param x       [tokens, in_features] input activations.
/// \param weight  [out_features, in_features] projection weights.
/// \param out     [tokens, out_features] output, same dtype as the inputs.
/// \return        OK, or InvalidArgument/Unimplemented for bad inputs.
Status Linear(ExecutionContext& ctx, const Tensor& x, const Tensor& weight, Tensor& out);

}  // namespace inferx::ops
