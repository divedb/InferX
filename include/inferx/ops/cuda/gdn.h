#ifndef INFERX_OPS_CUDA_GDN_H_
#define INFERX_OPS_CUDA_GDN_H_

#include "inferx/ops/op_context.h"
#include "inferx/ops/gdn.h"

namespace inferx::ops::cuda {

Status SplitGdnProjection(OpContext& ctx, const Tensor& packed, Tensor& conv_in,
                          Tensor& z, int64_t key_heads, int64_t key_dim, int64_t value_heads,
                          int64_t value_dim);
Status GdnCausalConv(OpContext& ctx, Tensor& x, const Tensor& weight,
                     const Tensor& state, const Tensor& batch_indices, const Tensor& qo_indptr,
                     int64_t kernel);
Status GdnGates(OpContext& ctx, const Tensor& ba, const Tensor& a_log,
                const Tensor& dt_bias, Tensor& beta, Tensor& g);
Status GdnRecurrent(OpContext& ctx, const Tensor& conv, int64_t query_width,
                    const Tensor& beta, const Tensor& g, Tensor& state,
                    const Tensor& slot_indices, const Tensor& qo_indptr,
                    const Tensor& batch_indices, Tensor& y);
Status RmsNormGated(OpContext& ctx, const Tensor& y, const Tensor& z,
                    const Tensor& weight, float eps, Tensor& out);

}  // namespace inferx::ops::cuda

#endif  // INFERX_OPS_CUDA_GDN_H_
