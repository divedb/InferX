/// \file
/// \brief Qwen2 checkpoint mapping and composition: standard HF names.
/// Qwen2/2.5 bias the Q/K/V projections but never the output projection.

#include "inferx/models/architectures/qwen2/model.h"

#include "inferx/models/layers/decoder.h"

namespace inferx::models::qwen2 {

const WeightNames& CheckpointNames() {
  static const WeightNames names{};
  return names;
}

const WeightLayout& CheckpointLayout() {
  static const WeightLayout layout{};
  return layout;
}

StatusOr<layers::StackConfig> BuildStackConfig(const Qwen2Config& config) {
  layers::AttentionConfig attention;
  attention.query_heads = config.num_attention_heads;
  attention.kv_heads = config.num_key_value_heads;
  attention.head_dim = config.head_dim;
  attention.qkv_bias = config.attention_bias;
  attention.rotary = config.rope.Params();

  layers::StackConfig stack;
  stack.hidden_size = config.hidden_size;
  stack.vocab_size = config.vocab_size;
  stack.final_norm = layers::NormConfig{static_cast<float>(config.rms_norm_eps)};
  stack.blocks.resize(config.num_hidden_layers);
  for (auto& block : stack.blocks) {
    block.norm = stack.final_norm;
    block.mixer = attention;
    block.feed_forward =
        layers::GatedMlpConfig{config.intermediate_size, config.activation};
  }
  return stack;
}

}  // namespace inferx::models::qwen2
