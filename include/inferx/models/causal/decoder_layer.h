#pragma once

#include <utility>
#include <variant>

#include "inferx/models/causal/model_traits.h"
#include "inferx/models/components/decoder_layer.h"
#include "inferx/models/components/gdn.h"
#include "inferx/models/components/mla.h"
#include "inferx/models/components/norm.h"
#include "inferx/models/diagnostic_trace.h"

namespace inferx::causal {

/// Owns one layer's weights. The stack lends activation workspace and the
/// runner lends persistent state. Topology is data: the mixer and
/// feed-forward variants dispatch to the shared runners (RunAttention,
/// RunGatedDeltaNet, RunFeedForward), so heterogeneous layers cost a
/// variant visit, not a new execution path. Under kPreNorm the final
/// residual addition is fused into the following layer's normalization (or
/// the stack's final normalization); under kOutputNorm each sublayer output
/// is normalized before joining the residual stream, so the same fusion
/// still holds.
template <ModelTraits Traits>
class DecoderLayer {
 public:
  DecoderLayer(const components::DecoderLayerConfig& config,
               components::DecoderLayerWeights weights)
      : input_norm_(config.norm, std::move(weights.input_norm)),
        ffn_norm_(config.norm, std::move(weights.post_mixer_norm)),
        mixer_out_norm_(MakeOutNorm(config.residual, config.mixer_out_norm,
                                    std::move(weights.mixer_out_norm))),
        feed_forward_out_norm_(MakeOutNorm(config.residual, config.feed_forward_out_norm,
                                           std::move(weights.feed_forward_out_norm))),
        mixer_(config.mixer),
        feed_forward_(config.feed_forward),
        mixer_weights_(std::move(weights.mixer)),
        feed_forward_weights_(std::move(weights.feed_forward)),
        norm_eps_(config.norm.eps),
        output_norm_residual_(config.residual == components::ResidualStyle::kOutputNorm) {
    static_assert(Traits::kNorm == NormPlacement::kPre);
  }

  using LayerState = std::variant<PagedKvState, RecurrentState>;
  Status Forward(bool first, Tensor& hidden, Tensor& normed, Tensor& mixed,
                 const AttentionBatch& batch, const LayerState& state, const KvBlockPool& pool,
                 const RecurrentStatePool& recurrent, components::AttentionWorkspace& attention_ws,
                 components::MlaWorkspace& mla_ws, components::GdnWorkspace& gdn_ws,
                 components::MlpWorkspace& mlp_ws, components::MoeWorkspace& moe_ws,
                 Tensor& packed, ops::ExecutionContext& ctx, DiagnosticTrace& trace,
                 const std::string& prefix) const {
    if (first) {
      INFERX_RETURN_IF_ERROR(input_norm_.Forward(ctx, hidden, normed));
    } else {
      INFERX_RETURN_IF_ERROR(input_norm_.AddForward(ctx, mixed, hidden, normed));
    }
    if (trace.enabled()) trace.Write(prefix + "input_norm", normed);
    if (auto* gdn = std::get_if<components::GdnWeights>(&mixer_weights_)) {
      INFERX_RETURN_IF_ERROR(components::RunGatedDeltaNet(
          std::get<components::GatedDeltaNetConfig>(mixer_), *gdn, normed, batch,
          std::get<RecurrentState>(state), recurrent, gdn_ws, ctx, &trace, prefix, mixed));
    } else if (auto* mla = std::get_if<components::MlaWeights>(&mixer_weights_)) {
      INFERX_RETURN_IF_ERROR(components::RunMlaAttention(
          std::get<components::MlaConfig>(mixer_), *mla, normed, norm_eps_, batch,
          std::get<PagedKvState>(state), pool, mla_ws, ctx, &trace, prefix, mixed));
    } else if (const auto* a = std::get_if<components::AttentionConfig>(&mixer_)) {
      INFERX_RETURN_IF_ERROR(
          components::RunAttention(*a, std::get<components::AttentionWeights>(mixer_weights_),
                                   normed, norm_eps_, batch, std::get<PagedKvState>(state), pool,
                                   attention_ws, &packed, ctx, &trace, prefix, mixed));
    } else {
      return InvalidArgumentError("layer mixer has no weights");
    }
    if (output_norm_residual_) {
      // Gemma sandwich: the mixer output is normalized before the residual
      // add, which the feed-forward-side norm then fuses.
      INFERX_RETURN_IF_ERROR(mixer_out_norm_->Forward(ctx, mixed, mixed));
      if (trace.enabled()) trace.Write(prefix + "mixer_out_norm", mixed);
    }
    INFERX_RETURN_IF_ERROR(ffn_norm_.AddForward(ctx, mixed, hidden, normed));
    if (trace.enabled()) trace.Write(prefix + "post_norm", normed);
    if (trace.enabled()) trace.Write(prefix + "residual", hidden);
    INFERX_RETURN_IF_ERROR(components::RunFeedForward(feed_forward_, feed_forward_weights_,
                                                      normed, mlp_ws, moe_ws, &packed, ctx, &trace,
                                                      prefix, mixed));
    if (output_norm_residual_) {
      INFERX_RETURN_IF_ERROR(feed_forward_out_norm_->Forward(ctx, mixed, mixed));
      if (trace.enabled()) trace.Write(prefix + "feed_forward_out_norm", mixed);
    }
    return OkStatus();
  }

 private:
  static std::optional<typename Traits::Norm> MakeOutNorm(
      components::ResidualStyle residual, const components::NormConfig& config,
      std::optional<Tensor> weight) {
    if (residual != components::ResidualStyle::kOutputNorm || !weight.has_value()) {
      return std::nullopt;
    }
    return typename Traits::Norm(config, std::move(*weight));
  }

  typename Traits::Norm input_norm_;
  typename Traits::Norm ffn_norm_;
  std::optional<typename Traits::Norm> mixer_out_norm_;
  std::optional<typename Traits::Norm> feed_forward_out_norm_;
  std::variant<components::AttentionConfig, components::MlaConfig,
               components::GatedDeltaNetConfig> mixer_;
  std::variant<components::SwiGluConfig, components::MoeConfig> feed_forward_;
  std::variant<components::AttentionWeights, components::MlaWeights,
               components::GdnWeights> mixer_weights_;
  std::variant<components::SwiGluWeights, components::MoeWeights> feed_forward_weights_;
  float norm_eps_;
  bool output_norm_residual_;
};

}  // namespace inferx::causal
