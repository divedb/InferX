#ifndef INFERX_OPS_CUDA_GATHER_H_
#define INFERX_OPS_CUDA_GATHER_H_

#include "inferx/core/status.h"
#include "inferx/core/tensor.h"
#include "inferx/ops/execution_context.h"
#include "inferx/ops/gather.h"

namespace inferx::ops::cuda {

Status GatherRows(ExecutionContext& ctx, const Tensor& src, const Tensor& indices, Tensor& out);

}  // namespace inferx::ops::cuda

#endif  // INFERX_OPS_CUDA_GATHER_H_
