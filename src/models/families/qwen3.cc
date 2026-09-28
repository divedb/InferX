/// \file
/// \brief Qwen3 traits and configuration translation for its variants.

#include <algorithm>
#include <string>
#include <vector>

#include "inferx/models/causal/build.h"
#include "inferx/models/causal/config_parser.h"
#include "inferx/models/components/decoder_layer.h"
#include "inferx/models/families/families.h"

namespace inferx::families {
namespace {

struct Qwen3Traits {
  static constexpr std::string_view kModelType = "qwen3";
  static constexpr std::string_view kArch = "Qwen3ForCausalLM";
  static constexpr causal::NormPlacement kNorm = causal::NormPlacement::kPre;
  static constexpr models::WeightNames kNames{};
  static constexpr models::WeightLayout kLayout{};
  using Norm = components::RmsNorm;
  using Attn =
      components::GqaAttention<components::QkvBias::kDisabled, components::QkNorm::kRmsNorm,
                               components::RopeStyle::kNeox>;
  using Mlp = components::GatedMlp<ops::Activation::kSilu>;
};

static_assert(causal::ModelTraits<Qwen3Traits>);

enum class Variant { kDense, kMoe, kNext };

/// Family identity selects the translator even when model_type is absent.
/// The common build path validates the resulting configuration.
template <Variant Kind>
StatusOr<causal::DecoderConfig> TranslateVariant(const nlohmann::json& j) {
  constexpr bool next = Kind == Variant::kNext;
  constexpr bool moe = Kind != Variant::kDense;
  INFERX_ASSIGN_OR_RETURN(auto config, causal::AttentionDecoderConfig(
                                            j, {/*qk_norm=*/Qwen3Traits::Attn::kQkNorm,
                                                /*plus_one_norm=*/next,
                                                /*qk_norm_plus_one=*/next}));
  const int64_t sparse_step = j.value("decoder_sparse_step", int64_t{1});
  if (sparse_step <= 0) return InvalidArgumentError("decoder_sparse_step must be positive");
  const auto dense_layers = j.value("mlp_only_layers", std::vector<int64_t>{});
  for (const auto layer : dense_layers) {
    if (layer < 0 || layer >= config.model.num_hidden_layers) {
      return InvalidArgumentError("mlp_only_layers index is out of range");
    }
  }
  components::MoeConfig experts;
  if (moe) {
    experts.num_experts = j.at("num_experts").get<int64_t>();
    experts.experts_per_token = j.at("num_experts_per_tok").get<int64_t>();
    experts.intermediate_size = j.at("moe_intermediate_size").get<int64_t>();
    experts.normalize_routing_weights = j.value("norm_topk_prob", true);
    experts.shared_intermediate_size = j.value("shared_expert_intermediate_size", int64_t{0});
    experts.gate_shared_expert = next && experts.shared_intermediate_size > 0;
  }
  std::vector<std::string> layer_types;
  if (j.contains("layer_types"))
    layer_types = j.at("layer_types").get<std::vector<std::string>>();
  if (j.contains("layer_types") && layer_types.size() != config.blocks.size()) {
    return InvalidArgumentError("layer_types must have one entry per decoder layer");
  }
  const int64_t interval = j.value("full_attention_interval", int64_t{4});
  if (next && interval <= 0)
    return InvalidArgumentError("full_attention_interval must be positive");
  for (size_t i = 0; i < config.blocks.size(); ++i) {
    auto& block = config.blocks[i];
    if (moe && (i + 1) % sparse_step == 0 &&
        std::find(dense_layers.begin(), dense_layers.end(), i) == dense_layers.end()) {
      block.feed_forward = experts;
    }
    const std::string layer_type =
        layer_types.empty()
            ? (next && (i + 1) % interval != 0 ? "linear_attention" : "full_attention")
            : layer_types[i];
    if (layer_type == "linear_attention" && next) {
      block.mixer =
          components::GatedDeltaNetConfig{j.at("linear_num_key_heads").get<int64_t>(),
                                          j.at("linear_num_value_heads").get<int64_t>(),
                                          j.at("linear_key_head_dim").get<int64_t>(),
                                          j.at("linear_value_head_dim").get<int64_t>(),
                                          j.at("linear_conv_kernel_dim").get<int64_t>()};
    } else if (layer_type == "full_attention") {
      if (next)
        std::get<components::AttentionConfig>(block.mixer).output_gate =
            components::OutputGate::kSigmoid;
    } else {
      return UnimplementedError("unsupported Qwen3 layer type: ", layer_type);
    }
  }
  return config;
}

}  // namespace

Family Qwen3() { return causal::MakeFamily<Qwen3Traits, TranslateVariant<Variant::kDense>>(); }

Family Qwen3Moe() {
  auto family = causal::MakeFamily<Qwen3Traits, TranslateVariant<Variant::kMoe>>();
  family.model_type = "qwen3_moe";
  family.architecture = "Qwen3MoeForCausalLM";
  return family;
}

Family Qwen3Next() {
  auto family = causal::MakeFamily<Qwen3Traits, TranslateVariant<Variant::kNext>>();
  family.model_type = "qwen3_next";
  family.architecture = "Qwen3NextForCausalLM";
  return family;
}

}  // namespace inferx::families
