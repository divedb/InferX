#ifndef INFERX_OPS_CUDA_GATHER_H_
#define INFERX_OPS_CUDA_GATHER_H_

#include "inferx/core/status.h"
#include "inferx/core/tensor.h"
#include "inferx/ops/execution_context.h"
#include "inferx/ops/gather.h"

namespace inferx::ops::cuda {

Status GatherRows(ExecutionContext& ctx, const Tensor& src, const Tensor& indices, Tensor& out);

Status GatherRowsRange(ExecutionContext& ctx, const Tensor& src, const Tensor& indices,
                       Tensor& out, int64_t row_begin);

Status CopyColumnBlock(ExecutionContext& ctx, const Tensor& src, Tensor& dst,
                       int64_t col_begin);

}  // namespace inferx::ops::cuda

#endif  // INFERX_OPS_CUDA_GATHER_H_
