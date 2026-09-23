#include "models/lm/config_parser.h"

#include <cmath>
#include <fstream>
#include <sstream>

namespace inferx::lm {

StatusOr<std::string> ReadConfig(const std::string& directory) {
  std::ifstream file(directory + "/config.json");
  if (!file) return NotFoundError("could not open config: ", directory);
  std::ostringstream text;
  text << file.rdbuf();
  return text.str();
}

StatusOr<DecoderConfig> AttentionDecoderConfig(const nlohmann::json& j,
                                               bool qk_norm, bool plus_one_norm) {
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
  layers::AttentionConfig attention;
  attention.query_heads = model.num_attention_heads;
  attention.kv_heads = model.num_key_value_heads;
  attention.head_dim = model.head_dim;
  attention.qk_norm = qk_norm;
  attention.projection_bias = j.value("attention_bias", false);
  attention.rotary.theta = model.rope_theta;
  nlohmann::json rope = nlohmann::json::object();
  for (const char* key : {"rope_scaling", "rope_parameters"}) {
    if (j.contains(key) && !j.at(key).is_null()) rope = j.at(key);
  }
  attention.rotary.type = rope.value("rope_type", rope.value("type", std::string("default")));
  attention.rotary.theta = rope.value("rope_theta", attention.rotary.theta);
  attention.rotary.factor = rope.value("factor", 1.0);
  attention.rotary.parameters_json = rope.dump();
  const double partial = rope.value("partial_rotary_factor", j.value("partial_rotary_factor", 1.0));
  if (!std::isfinite(partial) || partial <= 0 || partial > 1) {
    return InvalidArgumentError("partial_rotary_factor must be in (0, 1]");
  }
  attention.rotary.dim = static_cast<int64_t>(model.head_dim * partial);
  if (j.value("use_sliding_window", false)) {
    return UnimplementedError("sliding-window layer mapping is not implemented");
  }
  DecoderConfig config;
  config.model = model;
  config.final_norm = layers::NormConfig{model.rms_norm_eps, plus_one_norm};
  config.blocks.resize(model.num_hidden_layers);
  for (auto& block : config.blocks) {
    block.norm = config.final_norm;
    block.mixer = attention;
    block.feed_forward = layers::SwiGluConfig{model.intermediate_size};
  }
  return config;
}

}  // namespace inferx::lm
