#include "inferx/models/architectures/qwen3/model.h"
#include "inferx/models/layers/decoder.h"
#include "inferx/models/loading/weight_mapping.h"

namespace inferx::models::qwen3 {

/// Standard HuggingFace names; the generic loader fuses the separate
/// q/k/v projections into the packed row layout.
const WeightNames& CheckpointNames() {
  static const WeightNames names{};

  return names;
}

const WeightLayout& CheckpointLayout() {
  static const WeightLayout layout{};

  return layout;
}

StatusOr<layers::StackConfig> BuildStackConfig(const Qwen3Config& config) {
  layers::AttentionConfig attention;
  attention.query_heads = config.num_attention_heads;
  attention.kv_heads = config.num_key_value_heads;
  attention.head_dim = config.head_dim;
  attention.qk_norm = true;
  attention.qkv_bias = config.attention_bias;
  attention.output_bias = config.attention_bias;
  attention.rotary = config.rope.Params();

  layers::StackConfig stack;
  stack.hidden_size = config.hidden_size;
  stack.vocab_size = config.vocab_size;
  stack.final_norm = layers::NormConfig{static_cast<float>(config.rms_norm_eps)};
  stack.blocks.resize(config.num_hidden_layers);

  for (auto& block : stack.blocks) {
    block.norm = stack.final_norm;
    block.mixer = attention;
    block.feed_forward = layers::GatedMlpConfig{config.intermediate_size, config.activation};
  }

  return stack;
}

}  // namespace inferx::models::qwen3
