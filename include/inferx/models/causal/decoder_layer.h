#pragma once

#include <utility>

#include "inferx/models/causal/model_traits.h"
#include "inferx/models/components/decoder_layer.h"
#include "inferx/models/diagnostic_trace.h"

namespace inferx::causal {

/// Owns one layer's weights. The stack lends activation workspace and the
/// runner lends persistent state. The final residual addition is fused into
/// the following layer's normalization (or the stack's final normalization).
template <ModelTraits Traits>
class DecoderLayer {
 public:
  DecoderLayer(const components::DecoderLayerConfig& config,
               components::DecoderLayerWeights weights)
      : input_norm_(config.norm, std::move(weights.input_norm)),
        ffn_norm_(config.norm, std::move(weights.post_mixer_norm)),
        attention_(std::get<typename Traits::Attn::Config>(config.mixer),
                   std::move(weights.mixer)),
        mlp_(std::get<typename Traits::Mlp::Config>(config.feed_forward),
             std::get<typename Traits::Mlp::Weights>(std::move(weights.feed_forward))),
        norm_eps_(config.norm.eps) {
    static_assert(Traits::kNorm == NormPlacement::kPre);
  }

  Status Forward(bool first, Tensor& hidden, Tensor& normed, Tensor& mixed,
                 const AttentionBatch& batch, const PagedKvState& state,
                 const KvBlockPool& pool, components::AttentionWorkspace& attention_ws,
                 components::MlpWorkspace& mlp_ws, Tensor& packed, ops::ExecutionContext& ctx,
                 DiagnosticTrace& trace, const std::string& prefix) const {
    if (first) {
      INFERX_RETURN_IF_ERROR(input_norm_.Forward(ctx, hidden, normed));
    } else {
      INFERX_RETURN_IF_ERROR(input_norm_.AddForward(ctx, mixed, hidden, normed));
    }
    if (trace.enabled()) trace.Write(prefix + "input_norm", normed);
    INFERX_RETURN_IF_ERROR(attention_.Forward(normed, norm_eps_, batch, state, pool,
                                              attention_ws, &packed, ctx, &trace, prefix,
                                              mixed));
    INFERX_RETURN_IF_ERROR(ffn_norm_.AddForward(ctx, mixed, hidden, normed));
    if (trace.enabled()) trace.Write(prefix + "post_norm", normed);
    if (trace.enabled()) trace.Write(prefix + "residual", hidden);
    return mlp_.Forward(normed, mlp_ws, &packed, ctx, &trace, prefix, mixed);
  }

 private:
  typename Traits::Norm input_norm_;
  typename Traits::Norm ffn_norm_;
  typename Traits::Attn attention_;
  typename Traits::Mlp mlp_;
  float norm_eps_;
};

}  // namespace inferx::causal
