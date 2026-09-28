#ifndef INFERX_OPS_CUDA_LINEAR_H_
#define INFERX_OPS_CUDA_LINEAR_H_

#include "inferx/core/status.h"
#include "inferx/core/tensor.h"
#include "inferx/ops/execution_context.h"
#include "inferx/ops/linear.h"

namespace inferx::ops::cuda {

Status Linear(ExecutionContext& ctx, const Tensor& x, const Tensor& weight, Tensor& out);

}  // namespace inferx::ops::cuda

#endif  // INFERX_OPS_CUDA_LINEAR_H_
