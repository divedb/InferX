#ifndef INFERX_OPS_CUDA_ATTENTION_H_
#define INFERX_OPS_CUDA_ATTENTION_H_

#include "inferx/core/status.h"
#include "inferx/core/tensor.h"
#include "inferx/ops/attention.h"
#include "inferx/ops/execution_context.h"

namespace inferx::ops::cuda {

Status WritePagedKv(ExecutionContext& ctx, const Tensor& k, const Tensor& v,
                    const Tensor& positions, const Tensor& batch_indices,
                    const Tensor& kv_indptr, const Tensor& kv_indices,
                    const Tensor& key_cache, const Tensor& value_cache, int64_t block_size);

}  // namespace inferx::ops::cuda

#endif  // INFERX_OPS_CUDA_ATTENTION_H_
