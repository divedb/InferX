#ifndef INFERX_OPS_CUDA_MLA_H_
#define INFERX_OPS_CUDA_MLA_H_

#include "inferx/core/status.h"
#include "inferx/core/tensor.h"
#include "inferx/ops/op_context.h"

namespace inferx::ops::cuda {

Status AssembleMlaCaches(OpContext& ctx, const Tensor& k_rope, const Tensor& up_projected,
                         int64_t heads, int64_t nope, int64_t rope, int64_t v_dim, Tensor& k_out,
                         Tensor& v_out);

}  // namespace inferx::ops::cuda

#endif  // INFERX_OPS_CUDA_MLA_H_
