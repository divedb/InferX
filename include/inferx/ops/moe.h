#pragma once

#include <cstdint>

#include "inferx/core/status.h"
#include "inferx/core/tensor.h"
#include "inferx/ops/op_context.h"

namespace inferx::ops {

/// \brief How a MoE layer selects and weights its experts.
enum class RouterScoring : uint8_t {
  /// Softmax restricted to the top-k logits, renormalized over the k
  /// selection (Mixtral, Qwen3-MoE/Next, gpt-oss; a router bias shifts
  /// the logits before selection).
  kSoftmaxTopkRenorm,
  /// Sigmoid scores with DeepSeek's grouped selection and correction bias:
  /// the bias and group mask pick the experts, the unbiased sigmoid scores
  /// weight them, then optional renormalization and the routed scale.
  kSigmoidGroupTopk,
};

/// \brief Routing policy of one MoE layer.
struct RoutingConfig {
  RouterScoring scoring = RouterScoring::kSoftmaxTopkRenorm;
  int64_t topk = 0;             ///< Experts activated per token.
  bool normalize = true;        ///< Renormalize the selected weights.
  float routing_scale = 1.0f;   ///< Multiplier after selection (DeepSeek).
  int64_t group_count = 1;      ///< DeepSeek expert groups; 1 disables.
  int64_t group_topk = 0;       ///< Groups kept per token.
};

/// \brief Routes `hidden` ([rows, hidden]) through one router.
///
/// One kernel per row computes fp32 logits (weight, optional bias), applies
/// the scoring policy, and emits `topk_indices` ([rows, topk] int32) and
/// `topk_weights` ([rows, topk] float32). `correction_bias`
/// ([num_experts], float32) participates in DeepSeek selection only.
Status RouteTokens(OpContext& ctx, const Tensor& hidden, const Tensor& router_weight,
                   const Tensor* router_bias, const Tensor* correction_bias,
                   const RoutingConfig& config, Tensor& topk_indices, Tensor& topk_weights);

/// \brief Builds the expert dispatch for one step.
///
/// Histograms `topk_indices` into per-expert counts, copies the counts to
/// the host, and fills `token_rows`/`weights_by_slot` so slot
/// `offsets[expert] + i` holds the i-th (token row, weight) pair assigned to
/// that expert. Returns the host-side offsets, one per expert plus the end.
StatusOr<std::vector<int64_t>> BuildExpertDispatch(
    OpContext& ctx, const Tensor& topk_indices, const Tensor& topk_weights,
    int64_t num_experts, Tensor& counts, Tensor& offsets, Tensor& cursor, Tensor& token_rows,
    Tensor& weights_by_slot);

/// \brief Gathers rows for expert execution: out[i] <- hidden[token_rows[i]].
Status GatherRoutedTokens(OpContext& ctx, const Tensor& hidden, const Tensor& token_rows,
                          Tensor& out);

/// \brief Accumulates expert outputs back: out[r] += w * expert_rows[slot].
///
/// `token_rows` and `weights_by_slot` come from BuildExpertDispatch; the
/// expert's slice of `expert_rows` starts at its offset.
Status ScatterRoutedOutputs(OpContext& ctx, const Tensor& expert_rows,
                            const Tensor& token_rows, const Tensor& weights_by_slot,
                            int64_t begin, int64_t count, Tensor& out);

}  // namespace inferx::ops
