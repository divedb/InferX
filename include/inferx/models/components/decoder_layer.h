/// \file
/// \brief Per-layer configuration and canonical weights for checkpoint
/// translation. Execution topology lives in causal::DecoderLayer<Traits>;
/// variants also describe families whose execution is not implemented yet.

#ifndef INFERX_MODELS_COMPONENTS_DECODER_LAYER_H_
#define INFERX_MODELS_COMPONENTS_DECODER_LAYER_H_

#include <variant>

#include "inferx/models/components/attention.h"
#include "inferx/models/components/mlp.h"
#include "inferx/models/components/moe.h"
#include "inferx/models/components/moe.h"
#include "inferx/models/components/norm.h"

namespace inferx {

class DiagnosticTrace;

namespace components {

/// \brief Where a layer's normalization sits relative to its residual adds.
enum class ResidualStyle {
  /// Llama lineage: normalize the residual stream before each sublayer,
  /// add the sublayer output raw (fused into the next AddForward).
  kPreNorm,
  /// Gemma lineage: normalize each sublayer OUTPUT before it joins the
  /// residual stream (post_attention/pre_feedforward/post_feedforward norms).
  kOutputNorm,
};

/// \brief Configuration for one decoder layer.
struct DecoderLayerConfig {
  NormConfig norm;  ///< Applied before the mixer and before the feed-forward.
  /// Output-side norms applied to the mixer and feed-forward results; read
  /// only when `residual` is kOutputNorm.
  NormConfig mixer_out_norm;
  NormConfig feed_forward_out_norm;
  ResidualStyle residual = ResidualStyle::kPreNorm;
  std::variant<AttentionConfig, GatedDeltaNetConfig> mixer;
  std::variant<SwiGluConfig, MoeConfig> feed_forward;
};

/// \brief Weights for one decoder layer.
struct DecoderLayerWeights {
  Tensor input_norm;       ///< [hidden] mixer-side norm.
  Tensor post_mixer_norm;  ///< [hidden] feed-forward-side norm.
  /// Output-side norms (Gemma sandwich); absent when residual is kPreNorm.
  std::optional<Tensor> mixer_out_norm;
  std::optional<Tensor> feed_forward_out_norm;
  AttentionWeights mixer;  ///< Recurrent mixers are not loadable yet.
  std::variant<SwiGluWeights, MoeWeights> feed_forward;
};

/// \brief Runs whichever feed-forward the layer configures into `mixed_out`.
///
/// Dispatches SwiGLU and MoE; the hot path stays a direct call.
Status RunFeedForward(const std::variant<SwiGluConfig, MoeConfig>& config,
                      const std::variant<SwiGluWeights, MoeWeights>& weights,
                      const Tensor& normed, MlpWorkspace& mlp_ws, MoeWorkspace& moe_ws,
                      Tensor* packed_buffer, ops::ExecutionContext& ctx, DiagnosticTrace* trace,
                      std::string_view prefix, Tensor& mixed_out);

}  // namespace inferx::components
}  // namespace inferx

#endif  // INFERX_MODELS_COMPONENTS_DECODER_LAYER_H_
