/// \file
/// \brief Top-k routed mixture of gated-MLP experts with an optional shared
///        expert. The routing policy selects between renormalized-softmax
///        and sigmoid/grouped scoring; experts share the gated-MLP
///        execution path.

#ifndef INFERX_MODELS_LAYERS_MOE_H_
#define INFERX_MODELS_LAYERS_MOE_H_

#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

#include "inferx/core/status.h"
#include "inferx/core/tensor.h"
#include "inferx/models/layers/gated_mlp.h"
#include "inferx/ops/moe.h"
#include "inferx/ops/op_context.h"

namespace inferx::models::layers {

/// \brief Options of one routed-mixture feed-forward layer.
struct MoeConfig {
  int64_t num_experts = 0;
  int64_t experts_per_token = 0;  ///< Top-k.
  int64_t intermediate_size = 0;  ///< Per-expert gate/up width.
  ops::Activation activation = ops::Activation::kSilu;  ///< Expert activation.
  float oai_alpha = 1.702f;  ///< gpt-oss expert activation.
  float oai_limit = 7.0f;    ///< gpt-oss clamp.
  ops::RoutingConfig routing;  ///< Selection and weighting policy.
  bool has_router_bias = false;       ///< gpt-oss router carries a bias.
  bool has_correction_bias = false;   ///< DeepSeek e_score_correction_bias.
  /// gpt-oss stores all experts fused, MXFP4-quantized with interleaved
  /// gate/up rows; dequantized and de-interleaved at load.
  bool fused_mxfp4_experts = false;
  int64_t shared_intermediate_size = 0;  ///< 0 disables the shared expert.
  bool gate_shared_expert = false;  ///< Shared expert output sigmoid-gated.
};

/// \brief Router, routed experts, and optional shared expert.
struct MoeWeights {
  Tensor router;  ///< [num_experts, hidden]
  std::optional<Tensor> router_bias;  ///< [num_experts]; gpt-oss.
  /// [num_experts] float32; DeepSeek selection-only correction.
  std::optional<Tensor> correction_bias;
  std::vector<GatedMlpWeights> experts;  ///< One per routed expert.
  std::optional<GatedMlpWeights> shared_expert;  ///< Absent when disabled.
  std::optional<Tensor> shared_expert_gate;  ///< [1, hidden]; present only when gated.
};

/// \brief MoE workspace, sized once by the decoder stack. Buffers are
///        indexed by dispatch slot: slot `offset[e] + i` holds the i-th
///        token routed to expert e, so gather, expert GEMMs, and the
///        weighted scatter all address the same rows.
struct MoeWorkspace {
  std::optional<Tensor> topk_indices;    ///< [max_tokens * topk] int32
  std::optional<Tensor> topk_weights;    ///< [max_tokens * topk] float32
  std::optional<Tensor> counts;          ///< [num_experts] int32
  std::optional<Tensor> offsets;         ///< [num_experts + 1] int32
  std::optional<Tensor> cursor;          ///< [num_experts] int32 scratch
  std::optional<Tensor> token_rows;      ///< [max_tokens * topk] int32
  std::optional<Tensor> weights_by_slot; ///< [max_tokens * topk] float32
  std::optional<Tensor> gathered;        ///< [max_tokens * topk, hidden] bf16
  std::optional<Tensor> activated;       ///< [max_tokens * topk, inter] bf16
  std::optional<Tensor> expert_rows;     ///< [max_tokens * topk, hidden] bf16
  std::optional<Tensor> packed_gate_up;  ///< [max_tokens * topk, 2*inter] bf16
  std::optional<Tensor> shared_out;      ///< [max_tokens, hidden] bf16
  std::optional<Tensor> shared_gate;     ///< [max_tokens, 1] bf16 gate logits
};

/// \brief The single home for expert execution: route, dispatch, per-expert
///        GEMMs through the shared gated-MLP ops, weighted scatter, and the
///        optional gated shared expert.
Status RunMoe(const MoeConfig& config, const MoeWeights& weights, const Tensor& normed,
              MoeWorkspace& ws, GatedMlpWorkspace& mlp_ws, Tensor* packed_buffer,
              ops::OpContext& ctx, Tensor& mixed_out);

}  // namespace inferx::models::layers

#endif  // INFERX_MODELS_LAYERS_MOE_H_
