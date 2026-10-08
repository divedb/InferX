/// \file
/// \brief Mixtral checkpoint mapping and composition: every layer routes
/// through num_local_experts experts of the model intermediate width.

#include "inferx/models/architectures/mixtral/model.h"

#include "inferx/models/layers/decoder.h"

namespace inferx::models::mixtral {

const WeightNames& CheckpointNames() {
  static const WeightNames names{};
  return names;
}

const WeightLayout& CheckpointLayout() {
  static const WeightLayout layout{};
  return layout;
}

StatusOr<layers::StackConfig> BuildStackConfig(const MixtralConfig& config) {
  layers::AttentionConfig attention;
  attention.query_heads = config.num_attention_heads;
  attention.kv_heads = config.num_key_value_heads;
  attention.head_dim = config.head_dim;
  attention.qkv_bias = config.attention_bias;
  attention.output_bias = config.attention_bias;
  attention.rotary = config.rope.Params();
  attention.sliding_window =
      config.use_sliding_window && config.sliding_window > 0 ? config.sliding_window : 0;

  layers::MoeConfig experts;
  experts.num_experts = config.num_local_experts;
  experts.experts_per_token = config.num_experts_per_tok;
  experts.intermediate_size = config.intermediate_size;
  experts.activation = config.activation;
  experts.oai_limit = config.swiglu_limit;
  experts.routing.normalize = config.norm_topk_prob;

  layers::StackConfig stack;
  stack.hidden_size = config.hidden_size;
  stack.vocab_size = config.vocab_size;
  stack.final_norm = layers::NormConfig{static_cast<float>(config.rms_norm_eps)};
  stack.blocks.resize(config.num_hidden_layers);
  for (auto& block : stack.blocks) {
    block.norm = stack.final_norm;
    block.mixer = attention;
    block.feed_forward = experts;
  }
  return stack;
}

}  // namespace inferx::models::mixtral
