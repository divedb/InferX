#include "inferx/models/components/moe.h"

#include <vector>

#include "inferx/core/device_runtime.h"
#include "inferx/core/shape.h"
#include "inferx/ops/elementwise.h"
#include "inferx/ops/linear.h"
#include "inferx/ops/moe.h"

namespace inferx::components {

Status RunMoe(const MoeConfig& config, const MoeWeights& weights, const Tensor& normed,
              MoeWorkspace& ws, MlpWorkspace& mlp_ws, Tensor* packed_buffer,
              ops::OpContext& ctx, Tensor& mixed_out) {
  const int64_t rows = normed.Dim(0);
  const int64_t hidden = normed.Dim(1);

  const int64_t topk = config.experts_per_token;
  INFERX_ASSIGN_OR_RETURN(Tensor indices_flat, ws.topk_indices->Slice(0, rows * topk));
  INFERX_ASSIGN_OR_RETURN(Tensor indices2d, indices_flat.Reshape(Shape({rows, topk})));
  ops::RoutingConfig routing = config.routing;
  routing.topk = topk;
  INFERX_ASSIGN_OR_RETURN(Tensor weights2d, ws.topk_weights->Slice(0, rows * topk));
  INFERX_RETURN_IF_ERROR(ops::RouteTokens(
      ctx, normed, weights.router, weights.router_bias.has_value() ? &*weights.router_bias : nullptr,
      weights.correction_bias.has_value() ? &*weights.correction_bias : nullptr, routing,
      indices_flat, weights2d));

  // Route, then bring the per-expert counts to the host once: expert GEMMs
  // need host-side row counts. A fused MoE kernel removes this sync later.
  INFERX_ASSIGN_OR_RETURN(Tensor token_rows, ws.token_rows->Slice(0, rows * topk));
  INFERX_ASSIGN_OR_RETURN(Tensor weights_by_slot, ws.weights_by_slot->Slice(0, rows * topk));
  auto dispatch = ops::BuildExpertDispatch(ctx, indices_flat, weights2d, config.num_experts,
                                           *ws.counts, *ws.offsets, *ws.cursor, token_rows,
                                           weights_by_slot);
  INFERX_RETURN_IF_ERROR(dispatch.status());
  const std::vector<int64_t>& offsets = *dispatch;

  INFERX_RETURN_IF_ERROR(ops::GatherRoutedTokens(ctx, normed, token_rows, *ws.gathered));

  // Every expert runs the same gated-MLP ops over its slice of the gathered
  // tokens; the shared execution below is the only expert loop.
  for (int64_t e = 0; e < config.num_experts; ++e) {
    const int64_t count = offsets[e + 1] - offsets[e];
    if (count == 0) continue;
    INFERX_ASSIGN_OR_RETURN(auto x, ws.gathered->Slice(offsets[e], offsets[e + 1]));
    INFERX_ASSIGN_OR_RETURN(auto packed_flat,
                            ws.packed_gate_up->Slice(0, count * 2 * config.intermediate_size));
    INFERX_ASSIGN_OR_RETURN(auto packed,
                            packed_flat.Reshape(Shape({count, 2 * config.intermediate_size})));
    INFERX_RETURN_IF_ERROR(ops::Linear(ctx, x, weights.experts[e].packed_gate_up, packed));
    if (weights.experts[e].packed_bias.has_value()) {
      INFERX_RETURN_IF_ERROR(ops::AddBias(ctx, packed, *weights.experts[e].packed_bias, packed));
    }
    INFERX_ASSIGN_OR_RETURN(auto act_flat, ws.activated->Slice(0, count * config.intermediate_size));
    INFERX_ASSIGN_OR_RETURN(auto act, act_flat.Reshape(Shape({count, config.intermediate_size})));
    INFERX_RETURN_IF_ERROR(ops::PackedGatedActivation(ctx, packed, act, config.activation,
                                                      config.oai_alpha, config.oai_limit));
    INFERX_ASSIGN_OR_RETURN(auto out_rows, ws.expert_rows->Slice(offsets[e], offsets[e + 1]));
    INFERX_RETURN_IF_ERROR(ops::Linear(ctx, act, weights.experts[e].down.weight, out_rows));
    if (weights.experts[e].down_bias.has_value()) {
      INFERX_RETURN_IF_ERROR(ops::AddBias(ctx, out_rows, *weights.experts[e].down_bias, out_rows));
    }
  }

  // The layer output starts at the gated shared expert (when present), then
  // accumulates the routed experts' weighted rows.
  if (weights.shared_expert.has_value()) {
    SwiGluConfig shared_cfg;
    shared_cfg.intermediate_size = config.shared_intermediate_size;
    INFERX_RETURN_IF_ERROR(RunSwiGlu(shared_cfg, *weights.shared_expert, normed, mlp_ws,
                                     packed_buffer, ctx, mixed_out));
    if (weights.shared_expert_gate.has_value()) {
      // The gate is a per-token scalar: project, then scale sigmoid-wise.
      INFERX_ASSIGN_OR_RETURN(Tensor gate_rows, ws.shared_gate->Slice(0, rows));
      INFERX_ASSIGN_OR_RETURN(Tensor gate_2d, gate_rows.Reshape(Shape({rows, 1})));
      INFERX_RETURN_IF_ERROR(ops::Linear(ctx, normed, *weights.shared_expert_gate, gate_2d));
      INFERX_RETURN_IF_ERROR(ops::MulSigmoidRowGate(ctx, mixed_out, gate_2d));
    }
  } else {
    INFERX_RETURN_IF_ERROR(ops::MulScalar(ctx, mixed_out, 0.0f));
  }
  for (int64_t e = 0; e < config.num_experts; ++e) {
    const int64_t count = offsets[e + 1] - offsets[e];
    if (count == 0) continue;
    INFERX_RETURN_IF_ERROR(ops::ScatterRoutedOutputs(ctx, *ws.expert_rows, token_rows,
                                                     weights_by_slot, offsets[e], count,
                                                     mixed_out));
  }
  return OkStatus();
}

}  // namespace inferx::components
