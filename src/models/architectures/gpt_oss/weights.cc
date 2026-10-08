/// \file
/// \brief gpt-oss checkpoint mapping and composition: every layer routes
/// through fused MXFP4 experts (dequantized and de-interleaved at load),
/// attention sinks are on, and the schedule may window some layers.

#include "inferx/models/architectures/gpt_oss/model.h"

#include "inferx/models/layers/decoder.h"

namespace inferx::models::gpt_oss {

const WeightNames& CheckpointNames() {
  static const WeightNames names{};
  return names;
}

const WeightLayout& CheckpointLayout() {
  static const WeightLayout layout{};
  return layout;
}

StatusOr<layers::StackConfig> BuildStackConfig(const GptOssConfig& config) {
  layers::AttentionConfig attention;
  attention.query_heads = config.num_attention_heads;
  attention.kv_heads = config.num_key_value_heads;
  attention.head_dim = config.head_dim;
  // gpt-oss biases the Q/K/V projections and adds per-head sinks.
  attention.qkv_bias = true;
  attention.rotary = config.rope.Params();
  attention.sinks = true;

  layers::MoeConfig experts;
  experts.num_experts = config.num_local_experts;
  experts.experts_per_token = config.num_experts_per_tok;
  experts.intermediate_size = config.expert_intermediate_size;
  experts.activation = config.activation;
  experts.oai_alpha = 1.702f;
  experts.oai_limit = config.swiglu_limit;
  experts.has_router_bias = true;
  experts.fused_mxfp4_experts = true;

  layers::StackConfig stack;
  stack.hidden_size = config.hidden_size;
  stack.vocab_size = config.vocab_size;
  stack.final_norm = layers::NormConfig{static_cast<float>(config.rms_norm_eps)};
  stack.blocks.resize(config.num_hidden_layers);
  for (size_t i = 0; i < stack.blocks.size(); ++i) {
    auto& block = stack.blocks[i];
    block.norm = stack.final_norm;
    layers::AttentionConfig layer = attention;
    layer.sliding_window =
        (!config.layer_types.empty() && config.layer_types[i] == "sliding_attention")
            ? config.sliding_window
            : 0;
    block.mixer = layer;
    block.feed_forward = experts;
  }
  return stack;
}

}  // namespace inferx::models::gpt_oss
