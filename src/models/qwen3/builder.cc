#include "inferx/models/qwen3/builder.h"

#include <algorithm>

#include "inferx/models/lm/loader.h"
#include "models/lm/config_parser.h"

namespace inferx::qwen3 {

StatusOr<lm::DecoderConfig> ParseConfig(const std::string& text) {
  try {
    const auto j = nlohmann::json::parse(text);
    const std::string type = j.value("model_type", std::string());
    const bool next = type == "qwen3_next";
    const bool moe = type == "qwen3_moe" || next;
    if (type != "qwen3" && !moe) {
      return UnimplementedError("no Qwen3 decoder mapping for model_type ", type);
    }
    INFERX_ASSIGN_OR_RETURN(auto config, lm::AttentionDecoderConfig(j, /*qk_norm=*/true,
                                                                    /*plus_one_norm=*/next));
    const int64_t sparse_step = j.value("decoder_sparse_step", int64_t{1});
    if (sparse_step <= 0) return InvalidArgumentError("decoder_sparse_step must be positive");
    const auto dense_layers = j.value("mlp_only_layers", std::vector<int64_t>{});
    for (const auto layer : dense_layers) {
      if (layer < 0 || layer >= config.model.num_hidden_layers) {
        return InvalidArgumentError("mlp_only_layers index is out of range");
      }
    }
    layers::MoeConfig experts;
    if (moe) {
      experts.num_experts = j.at("num_experts").get<int64_t>();
      experts.experts_per_token = j.at("num_experts_per_tok").get<int64_t>();
      experts.intermediate_size = j.at("moe_intermediate_size").get<int64_t>();
      experts.normalize_routing_weights = j.value("norm_topk_prob", true);
      experts.shared_intermediate_size = j.value("shared_expert_intermediate_size", int64_t{0});
      experts.gate_shared_expert = next && experts.shared_intermediate_size > 0;
    }
    std::vector<std::string> layer_types;
    if (j.contains("layer_types")) layer_types = j.at("layer_types").get<std::vector<std::string>>();
    if (j.contains("layer_types") && layer_types.size() != config.blocks.size()) {
      return InvalidArgumentError("layer_types must have one entry per decoder layer");
    }
    const int64_t interval = j.value("full_attention_interval", int64_t{4});
    if (next && interval <= 0) return InvalidArgumentError("full_attention_interval must be positive");
    for (size_t i = 0; i < config.blocks.size(); ++i) {
      auto& block = config.blocks[i];
      if (moe && (i + 1) % sparse_step == 0 &&
          std::find(dense_layers.begin(), dense_layers.end(), i) == dense_layers.end()) {
        block.feed_forward = experts;
      }
      const std::string layer_type = layer_types.empty()
          ? (next && (i + 1) % interval != 0 ? "linear_attention" : "full_attention")
          : layer_types[i];
      if (layer_type == "linear_attention" && next) {
        block.mixer = layers::GatedDeltaNetConfig{
            j.at("linear_num_key_heads").get<int64_t>(),
            j.at("linear_num_value_heads").get<int64_t>(),
            j.at("linear_key_head_dim").get<int64_t>(),
            j.at("linear_value_head_dim").get<int64_t>(),
            j.at("linear_conv_kernel_dim").get<int64_t>()};
      } else if (layer_type == "full_attention") {
        if (next) std::get<layers::AttentionConfig>(block.mixer).output_gate = layers::OutputGate::kSigmoid;
      } else {
        return UnimplementedError("unsupported Qwen3 layer type: ", layer_type);
      }
    }
    INFERX_RETURN_IF_ERROR(config.Validate());
    return config;
  } catch (const nlohmann::json::exception& e) {
    return InvalidArgumentError("invalid Qwen3 config: ", e.what());
  }
}

StatusOr<std::unique_ptr<Model>> Load(const std::string& directory, DeviceId device,
                                      int max_tokens, int max_seqs, ops::AttentionBackend backend) {
  INFERX_ASSIGN_OR_RETURN(auto text, lm::ReadConfig(directory));
  INFERX_ASSIGN_OR_RETURN(auto config, ParseConfig(text));
  // Next's packed recurrent projection format needs its own checkpoint mapping.
  if (config.model.model_type == "qwen3_next") {
    return UnimplementedError("Qwen3-Next checkpoint mapping and recurrent execution are pending");
  }
  config.attention_backend = backend;
  return lm::LoadCausalLM(directory, config, {}, device, max_tokens, max_seqs);
}

}  // namespace inferx::qwen3
