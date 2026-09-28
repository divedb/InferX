#pragma once

#include <utility>

#include "inferx/models/causal/causal_lm.h"
#include "inferx/models/causal/config_parser.h"
#include "inferx/models/model_registry.h"

namespace inferx::causal {

template <ModelTraits Traits>
StatusOr<DecoderConfig> TranslateDenseConfig(const nlohmann::json& json) {
  static_assert(Traits::kNorm == NormPlacement::kPre);
  static_assert(Traits::Attn::kRope == components::RopeStyle::kNeox);
  INFERX_ASSIGN_OR_RETURN(auto config,
                          AttentionDecoderConfig(
                              json, {/*qk_norm=*/Traits::Attn::kQkNorm}));
  if constexpr (Traits::Attn::kQkvBias) {
    for (auto& block : config.blocks) {
      std::get<components::AttentionConfig>(block.mixer).qkv_bias = true;
    }
  }
  return config;
}

/// Instantiates a typed executor behind the serving interface. Translators
/// may describe heterogeneous models; unsupported blocks are rejected by
/// PrepareCausalLM before weights are read or concrete layers constructed.
template <ModelTraits Traits, auto Translate = TranslateDenseConfig<Traits>>
Family MakeFamily() {
  return Family{
      .model_type = Traits::kModelType,
      .architecture = Traits::kArch,
      .build = [](const nlohmann::json& json, models::LoadedCheckpoint& checkpoint,
                  DeviceId device, int max_tokens, int max_seqs,
                  const ParallelConfig& parallel) -> StatusOr<std::unique_ptr<Model>> {
        INFERX_ASSIGN_OR_RETURN(auto config, Translate(json));
        for (const auto& block : config.blocks) {
          if (const auto* attention = std::get_if<components::AttentionConfig>(&block.mixer)) {
            if (attention->qk_norm != Traits::Attn::kQkNorm) {
              return InvalidArgumentError(
                  "translated Q/K normalization disagrees with model traits");
            }
          }
        }
        INFERX_ASSIGN_OR_RETURN(
            auto prepared,
            PrepareCausalLM(checkpoint, std::move(config), Traits::kNames, Traits::kLayout,
                            device, max_tokens, max_seqs, parallel));
        return std::unique_ptr<Model>(
            new CausalLM<Traits>(std::move(prepared), max_tokens, max_seqs));
      },
      .translate = Translate};
}

}  // namespace inferx::causal
