/// \file
/// \brief Mixture-of-experts families: Mixtral, gpt-oss, and DeepSeek-V3.
///        Routing policy differs; expert execution is shared.

#include <string>
#include <vector>

#include "inferx/models/causal/build.h"
#include "inferx/models/causal/config_parser.h"
#include "inferx/models/components/decoder_layer.h"
#include "inferx/models/components/mla.h"
#include "inferx/models/families/families.h"

namespace inferx::families {
namespace {

struct MixtralTraits {
  static constexpr std::string_view kModelType = "mixtral";
  static constexpr std::string_view kArch = "MixtralForCausalLM";
  static constexpr causal::NormPlacement kNorm = causal::NormPlacement::kPre;
  static constexpr models::WeightNames kNames{};
  static constexpr models::WeightLayout kLayout{};
  using Norm = components::RmsNorm;
  using Attn =
      components::GqaAttention<components::QkvBias::kDisabled, components::QkNorm::kNone,
                               components::RopeStyle::kNeox>;
  using Mlp = components::GatedMlp<ops::Activation::kSilu>;
};

static_assert(causal::ModelTraits<MixtralTraits>);

struct GptOssTraits {
  static constexpr std::string_view kModelType = "gpt_oss";
  static constexpr std::string_view kArch = "GptOssForCausalLM";
  static constexpr causal::NormPlacement kNorm = causal::NormPlacement::kPre;
  static constexpr models::WeightNames kNames{};
  static constexpr models::WeightLayout kLayout{};
  using Norm = components::RmsNorm;
  using Attn =
      components::GqaAttention<components::QkvBias::kEnabled, components::QkNorm::kNone,
                               components::RopeStyle::kNeox>;
  using Mlp = components::GatedMlp<ops::Activation::kSiluOai>;
};

static_assert(causal::ModelTraits<GptOssTraits>);

struct DeepseekTraits {
  static constexpr std::string_view kModelType = "deepseek_v3";
  static constexpr std::string_view kArch = "DeepseekV3ForCausalLM";
  static constexpr causal::NormPlacement kNorm = causal::NormPlacement::kPre;
  static constexpr models::WeightNames kNames{};
  static constexpr models::WeightLayout kLayout{};
  using Norm = components::RmsNorm;
  using Attn =
      components::GqaAttention<components::QkvBias::kDisabled, components::QkNorm::kNone,
                               components::RopeStyle::kNeox>;
  using Mlp = components::GatedMlp<ops::Activation::kSilu>;
};

static_assert(causal::ModelTraits<DeepseekTraits>);

StatusOr<causal::DecoderConfig> TranslateMixtral(const nlohmann::json& j) {
  INFERX_ASSIGN_OR_RETURN(auto config, causal::AttentionDecoderConfig(j, {/*qk_norm=*/false}));
  const int64_t window = j.value("sliding_window", int64_t{0});
  components::MoeConfig experts;
  experts.num_experts = j.at("num_local_experts").get<int64_t>();
  experts.experts_per_token = j.at("num_experts_per_tok").get<int64_t>();
  experts.intermediate_size = config.model.intermediate_size;
  experts.routing.normalize = j.value("norm_topk_prob", true);
  for (auto& block : config.blocks) {
    std::get<components::AttentionConfig>(block.mixer).sliding_window =
        j.value("use_sliding_window", true) ? window : 0;
    block.feed_forward = experts;
  }
  return config;
}

StatusOr<causal::DecoderConfig> TranslateGptOss(const nlohmann::json& j) {
  INFERX_ASSIGN_OR_RETURN(auto config, causal::AttentionDecoderConfig(j, {/*qk_norm=*/false}));
  components::MoeConfig experts;
  experts.num_experts = j.at("num_local_experts").get<int64_t>();
  experts.experts_per_token = j.at("num_experts_per_tok").get<int64_t>();
  experts.intermediate_size = j.at("intermediate_size").get<int64_t>();
  experts.has_router_bias = true;
  experts.fused_mxfp4_experts = true;
  experts.activation = ops::Activation::kSiluOai;
  experts.oai_limit = j.value("swiglu_limit", 7.0);
  std::vector<std::string> layer_types;
  if (j.contains("layer_types")) {
    layer_types = j.at("layer_types").get<std::vector<std::string>>();
  }
  for (size_t i = 0; i < config.blocks.size(); ++i) {
    auto& a = std::get<components::AttentionConfig>(config.blocks[i].mixer);
    a.sinks = true;
    a.sliding_window =
        (!layer_types.empty() && layer_types[i] == "sliding_attention")
            ? j.value("sliding_window", int64_t{0})
            : 0;
    config.blocks[i].feed_forward = experts;
  }
  return config;
}

StatusOr<causal::DecoderConfig> TranslateDeepseekV3(const nlohmann::json& j) {
  INFERX_ASSIGN_OR_RETURN(auto config, causal::AttentionDecoderConfig(j, {/*qk_norm=*/false}));
  components::MlaConfig mla;
  mla.query_heads = config.model.num_attention_heads;
  mla.q_lora_rank = j.at("q_lora_rank").get<int64_t>();
  mla.kv_lora_rank = j.at("kv_lora_rank").get<int64_t>();
  mla.qk_nope_head_dim = j.at("qk_nope_head_dim").get<int64_t>();
  mla.qk_rope_head_dim = j.at("qk_rope_head_dim").get<int64_t>();
  mla.v_head_dim = j.at("v_head_dim").get<int64_t>();
  mla.rotary = std::get<components::AttentionConfig>(config.blocks.front().mixer).rotary;
  mla.rotary.dim = mla.qk_rope_head_dim;

  components::MoeConfig experts;
  experts.num_experts = j.at("n_routed_experts").get<int64_t>();
  experts.experts_per_token = j.at("num_experts_per_tok").get<int64_t>();
  experts.intermediate_size = j.at("moe_intermediate_size").get<int64_t>();
  experts.routing.scoring = ops::RouterScoring::kSigmoidGroupTopk;
  experts.routing.normalize = j.value("norm_topk_prob", true);
  experts.routing.routing_scale = j.value("routed_scaling_factor", 1.0);
  experts.routing.group_count = j.value("n_group", int64_t{1});
  experts.routing.group_topk = j.value("topk_group", int64_t{0});
  if (experts.routing.group_topk <= 0) experts.routing.group_topk = experts.routing.group_count;
  experts.has_correction_bias = true;

  const int64_t dense_prefix = j.value("first_k_dense_replace", int64_t{0});
  for (size_t i = 0; i < config.blocks.size(); ++i) {
    config.blocks[i].mixer = mla;
    if (static_cast<int64_t>(i) >= dense_prefix) config.blocks[i].feed_forward = experts;
  }
  return config;
}

}  // namespace

Family Mixtral() { return causal::MakeFamily<MixtralTraits, TranslateMixtral>(); }

Family GptOss() { return causal::MakeFamily<GptOssTraits, TranslateGptOss>(); }

Family DeepseekV3() { return causal::MakeFamily<DeepseekTraits, TranslateDeepseekV3>(); }

}  // namespace inferx::families
