#ifndef INFERX_OPS_CUDA_ROTARY_H_
#define INFERX_OPS_CUDA_ROTARY_H_

#include "inferx/core/status.h"
#include "inferx/core/tensor.h"
#include "inferx/ops/execution_context.h"
#include "inferx/ops/rotary.h"

namespace inferx::ops::cuda {
Status NormalizeAndApplyRope(ExecutionContext& ctx, const Tensor& q, const Tensor& k,
                             const Tensor& q_weight, const Tensor& k_weight,
                             const Tensor& positions, float eps, const RotaryParams& params);

Status ApplyRope(ExecutionContext& ctx, const Tensor& q, const Tensor& k,
                 const Tensor& positions, const RotaryParams& params);

}  // namespace inferx::ops::cuda

#endif  // INFERX_OPS_CUDA_ROTARY_H_
