#include "inferx/models/components/gdn.h"

#include <string_view>

#include "inferx/core/shape.h"
#include "inferx/models/model.h"
#include "inferx/ops/gdn.h"
#include "inferx/ops/linear.h"

namespace inferx::components {

Status RunGatedDeltaNet(const GatedDeltaNetConfig& a, const GdnWeights& w, const Tensor& normed,
                        const AttentionBatch& batch, const RecurrentState& state,
                        const RecurrentStatePool& pool, GdnWorkspace& ws,
                        ops::OpContext& ctx, Tensor& mixed_out) {
  const int64_t rows = normed.Dim(0);
  const int64_t kh = a.key_heads, vh = a.value_heads;
  const int64_t kd = a.key_dim, vd = a.value_dim;
  const int64_t q_total = kh * kd, v_total = vh * vd;

  INFERX_ASSIGN_OR_RETURN(Tensor packed,
                          ws.packed->Slice(0, rows * (2 * q_total + 2 * v_total))
                              .value()
                              .Reshape(Shape({rows, 2 * q_total + 2 * v_total})));
  INFERX_RETURN_IF_ERROR(ops::Linear(ctx, normed, w.in_proj_qkvz, packed));
  INFERX_ASSIGN_OR_RETURN(Tensor conv_in,
                          ws.conv_in->Slice(0, rows * (2 * q_total + v_total))
                              .value()
                              .Reshape(Shape({rows, 2 * q_total + v_total})));
  INFERX_ASSIGN_OR_RETURN(Tensor z, ws.z->Slice(0, rows * v_total).value().Reshape(
                                        Shape({rows, v_total})));
  INFERX_RETURN_IF_ERROR(
      ops::SplitGdnProjection(ctx, packed, conv_in, z, kh, kd, vh, vd));

  INFERX_ASSIGN_OR_RETURN(Tensor conv_state, pool.ConvState(state.pool_layer));
  INFERX_RETURN_IF_ERROR(ops::GdnCausalConv(ctx, conv_in, w.conv1d, conv_state,
                                            batch.batch_indices, batch.qo_indptr,
                                            a.conv_kernel_size));
  INFERX_ASSIGN_OR_RETURN(Tensor ba,
                          ws.ba->Slice(0, rows * 2 * vh).value().Reshape(Shape({rows, 2 * vh})));
  INFERX_RETURN_IF_ERROR(ops::Linear(ctx, normed, w.in_proj_ba, ba));
  INFERX_ASSIGN_OR_RETURN(Tensor beta, ws.beta->Slice(0, rows * vh).value().Reshape(
                                           Shape({rows, vh})));
  INFERX_ASSIGN_OR_RETURN(
      Tensor g, ws.g->Slice(0, rows * vh).value().Reshape(Shape({rows, vh})));
  INFERX_RETURN_IF_ERROR(ops::GdnGates(ctx, ba, w.a_log, w.dt_bias, beta, g));

  INFERX_ASSIGN_OR_RETURN(Tensor state_t, pool.State(state.pool_layer));
  INFERX_ASSIGN_OR_RETURN(Tensor y, ws.y->Slice(0, rows * v_total).value().Reshape(
                                        Shape({rows, v_total})));
  if (!batch.recurrent_indices.has_value()) {
    return InvalidArgumentError("recurrent layer ran without slot indices");
  }
  INFERX_RETURN_IF_ERROR(ops::GdnRecurrent(ctx, conv_in, q_total, beta, g, state_t,
                                           *batch.recurrent_indices, batch.qo_indptr,
                                           batch.batch_indices, y));
  // The gated norm consumes y and z per head.
  INFERX_RETURN_IF_ERROR(ops::RmsNormGated(ctx, y, z, w.norm_weight, 1e-6f, y));
  INFERX_RETURN_IF_ERROR(ops::Linear(ctx, y, w.out_proj, mixed_out));
  return OkStatus();
}

}  // namespace inferx::components
