/// \file
/// \brief Dense decoder families whose identity differs only in policy:
///        Qwen2.5 (biased QKV), Mistral (sliding window), and Gemma 3
///        (output norms, GeGLU, plus-one RMSNorm, embedding scale).

#include <cmath>
#include <string>

#include "inferx/models/causal/build.h"
#include "inferx/models/causal/config_parser.h"
#include "inferx/models/components/decoder_layer.h"
#include "inferx/models/families/families.h"

namespace inferx::families {
namespace {

struct Qwen2Traits {
  static constexpr std::string_view kModelType = "qwen2";
  static constexpr std::string_view kArch = "Qwen2ForCausalLM";
  static constexpr causal::NormPlacement kNorm = causal::NormPlacement::kPre;
  static constexpr models::WeightNames kNames{};
  static constexpr models::WeightLayout kLayout{};
  using Norm = components::RmsNorm;
  using Attn =
      components::GqaAttention<components::QkvBias::kEnabled, components::QkNorm::kNone,
                               components::RopeStyle::kNeox>;
  using Mlp = components::GatedMlp<ops::Activation::kSilu>;
};

static_assert(causal::ModelTraits<Qwen2Traits>);

struct MistralTraits {
  static constexpr std::string_view kModelType = "mistral";
  static constexpr std::string_view kArch = "MistralForCausalLM";
  static constexpr causal::NormPlacement kNorm = causal::NormPlacement::kPre;
  static constexpr models::WeightNames kNames{};
  static constexpr models::WeightLayout kLayout{};
  using Norm = components::RmsNorm;
  using Attn =
      components::GqaAttention<components::QkvBias::kDisabled, components::QkNorm::kNone,
                               components::RopeStyle::kNeox>;
  using Mlp = components::GatedMlp<ops::Activation::kSilu>;
};

static_assert(causal::ModelTraits<MistralTraits>);

struct Gemma3Traits {
  static constexpr std::string_view kModelType = "gemma3_text";
  static constexpr std::string_view kArch = "Gemma3ForCausalLM";
  static constexpr causal::NormPlacement kNorm = causal::NormPlacement::kPre;
  static constexpr models::WeightNames kNames{[] {
    models::WeightNames names{};
    // Gemma's sandwich: the mixer output is normalized before the residual
    // add, and the feed-forward reads its own pre-norm.
    names.attn_out_norm = "post_attention_layernorm";
    names.ffn_norm = "pre_feedforward_layernorm";
    names.ffn_out_norm = "post_feedforward_layernorm";
    return names;
  }()};
  static constexpr models::WeightLayout kLayout{};
  using Norm = components::RmsNorm;
  using Attn =
      components::GqaAttention<components::QkvBias::kEnabled, components::QkNorm::kRmsNorm,
                               components::RopeStyle::kNeox>;
  using Mlp = components::GatedMlp<ops::Activation::kGeluTanh>;
};

static_assert(causal::ModelTraits<Gemma3Traits>);

/// Gemma's attention logit scale comes from query_pre_attn_scalar.
StatusOr<causal::DecoderConfig> TranslateGemma3(const nlohmann::json& j) {
  INFERX_ASSIGN_OR_RETURN(auto config,
                          causal::AttentionDecoderConfig(
                              j, {/*qk_norm=*/true, /*plus_one_norm=*/true,
                                  /*qk_norm_plus_one=*/true}));
  config.embedding_scale = std::sqrt(static_cast<float>(config.model.hidden_size));
  const float pre_attn = j.value("query_pre_attn_scalar", 0.0f);
  const int64_t window = j.value("sliding_window", int64_t{0});
  const int64_t pattern = j.value("sliding_window_pattern", int64_t{6});
  const double rope_local = j.value("rope_local_base_freq", 10000.0);
  for (size_t i = 0; i < config.blocks.size(); ++i) {
    auto& block = config.blocks[i];
    block.residual = components::ResidualStyle::kOutputNorm;
    block.mixer_out_norm = block.norm;
    block.feed_forward_out_norm = block.norm;
    auto& a = std::get<components::AttentionConfig>(block.mixer);
    if (pre_attn > 0.0f) a.scale_override = 1.0f / std::sqrt(pre_attn);
    const bool global = pattern > 0 && (i + 1) % pattern == 0;
    components::RotaryConfig local;
    local.dim = a.rotary.dim;
    local.theta = static_cast<float>(rope_local);
    a.rotary = global ? a.rotary : local;
    a.sliding_window = global || window <= 0 ? 0 : window;
  }
  return config;
}

}  // namespace

StatusOr<causal::DecoderConfig> TranslateQwen2(const nlohmann::json& j) {
  INFERX_ASSIGN_OR_RETURN(auto config, causal::AttentionDecoderConfig(j, {/*qk_norm=*/false}));
  // Qwen2/2.5 bias the Q/K/V projections but never the output projection.
  for (auto& block : config.blocks) {
    std::get<components::AttentionConfig>(block.mixer).output_bias = false;
  }
  return config;
}

Family Qwen2() { return causal::MakeFamily<Qwen2Traits, TranslateQwen2>(); }

StatusOr<causal::DecoderConfig> TranslateMistral(const nlohmann::json& j) {
  INFERX_ASSIGN_OR_RETURN(auto config, causal::AttentionDecoderConfig(j, {/*qk_norm=*/false}));
  const int64_t window = j.value("sliding_window", int64_t{0});
  if (j.value("use_sliding_window", false) && window > 0) {
    for (auto& block : config.blocks) {
      std::get<components::AttentionConfig>(block.mixer).sliding_window = window;
    }
  }
  return config;
}

Family Mistral() { return causal::MakeFamily<MistralTraits, TranslateMistral>(); }

Family Gemma3() { return causal::MakeFamily<Gemma3Traits, TranslateGemma3>(); }

}  // namespace inferx::families
