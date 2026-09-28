#include "inferx/models/causal/config_parser.h"

#include <cmath>

namespace inferx::causal {

StatusOr<nlohmann::json> ParseConfigJson(std::string_view text) {
  try {
    auto json = nlohmann::json::parse(text);
    if (!json.is_object()) return InvalidArgumentError("config.json must be an object");
    return json;
  } catch (const nlohmann::json::exception& e) {
    return InvalidArgumentError("invalid config.json: ", e.what());
  }
}

StatusOr<DecoderConfig> AttentionDecoderConfig(const nlohmann::json& j, bool qk_norm,
                                               bool plus_one_norm) {
  INFERX_ASSIGN_OR_RETURN(auto model, CheckpointConfig::FromJson(j.dump()));
  if (j.contains("text_config")) {
    return UnimplementedError("multimodal wrappers require their own model builder");
  }
  if (j.contains("quantization_config") && !j.at("quantization_config").is_null()) {
    return UnimplementedError("quantized checkpoint mapping is not implemented");
  }
  if (model.hidden_act != "silu" || j.value("mlp_bias", false)) {
    return UnimplementedError("decoder requires bias-free SwiGLU feed-forward layers");
  }
  components::AttentionConfig attention;
  attention.query_heads = model.num_attention_heads;
  attention.kv_heads = model.num_key_value_heads;
  attention.head_dim = model.head_dim;
  attention.qk_norm = qk_norm;
  attention.qkv_bias = j.value("attention_bias", false);
  attention.output_bias = j.value("attention_bias", false);
  attention.rotary.theta = model.rope_theta;
  nlohmann::json rope = nlohmann::json::object();
  for (const char* key : {"rope_scaling", "rope_parameters"}) {
    if (j.contains(key) && !j.at(key).is_null()) rope = j.at(key);
  }
  components::RotaryConfig& rotary = attention.rotary;
  rotary.type = rope.value("rope_type", rope.value("type", std::string("default")));
  rotary.theta = rope.value("rope_theta", rotary.theta);
  rotary.factor = rope.value("factor", 1.0);
  rotary.parameters_json = rope.dump();
  if (rotary.type == "llama3") {
    rotary.low_freq_factor = rope.value("low_freq_factor", 1.0);
    rotary.high_freq_factor = rope.value("high_freq_factor", 4.0);
    rotary.original_max_position =
        rope.value("original_max_position_embeddings", model.max_position_embeddings);
  } else if (rotary.type == "yarn") {
    rotary.beta_fast = rope.value("beta_fast", 32.0);
    rotary.beta_slow = rope.value("beta_slow", 1.0);
    rotary.truncate = rope.value("truncate", true);
    const int64_t original =
        rope.value("original_max_position_embeddings", int64_t{0});
    if (original > 0) {
      rotary.original_max_position = original;
      // The effective factor is the context ratio once the original training
      // length is known (reference rule; matches the config's own factor).
      rotary.factor =
          static_cast<double>(model.max_position_embeddings) / static_cast<double>(original);
    }
  }
  const double partial =
      rope.value("partial_rotary_factor", j.value("partial_rotary_factor", 1.0));
  if (!std::isfinite(partial) || partial <= 0 || partial > 1) {
    return InvalidArgumentError("partial_rotary_factor must be in (0, 1]");
  }
  rotary.dim = static_cast<int64_t>(model.head_dim * partial);
  if (j.value("use_sliding_window", false)) {
    return UnimplementedError("sliding-window layer mapping is not implemented");
  }
  DecoderConfig config;
  config.model = model;
  config.final_norm = components::NormConfig{model.rms_norm_eps, plus_one_norm};
  config.blocks.resize(model.num_hidden_layers);
  for (auto& block : config.blocks) {
    block.norm = config.final_norm;
    block.mixer = attention;
    block.feed_forward = components::SwiGluConfig{model.intermediate_size};
  }
  return config;
}

}  // namespace inferx::causal
