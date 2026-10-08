/// \file
/// \brief Gemma3 checkpoint mapping and composition. Gemma's sandwich:
/// the mixer output is normalized before the residual add, and the
/// feed-forward reads its own pre-norm, so the tensor names differ from
/// the Llama lineage. Local (windowed) layers use their own rope base
/// frequency; embedding activations scale by sqrt(hidden).

#include <cmath>

#include "inferx/models/architectures/gemma3/model.h"

#include "inferx/models/layers/decoder.h"

namespace inferx::models::gemma3 {

const WeightNames& CheckpointNames() {
  static const WeightNames names{[] {
    WeightNames names{};
    names.attn_out_norm = "post_attention_layernorm";
    names.ffn_norm = "pre_feedforward_layernorm";
    names.ffn_out_norm = "post_feedforward_layernorm";
    return names;
  }()};
  return names;
}

const WeightLayout& CheckpointLayout() {
  static const WeightLayout layout{};
  return layout;
}

StatusOr<layers::StackConfig> BuildStackConfig(const Gemma3Config& config) {
  layers::AttentionConfig attention;
  attention.query_heads = config.num_attention_heads;
  attention.kv_heads = config.num_key_value_heads;
  attention.head_dim = config.head_dim;
  attention.qk_norm = true;
  attention.qk_norm_plus_one = true;
  attention.qkv_bias = true;
  attention.rotary = config.rope.Params();
  if (config.query_pre_attn_scalar > 0.0) {
    attention.scale_override =
        static_cast<float>(1.0 / std::sqrt(config.query_pre_attn_scalar));
  }

  const layers::NormConfig norm{static_cast<float>(config.rms_norm_eps),
                                 /*plus_one=*/true};

  layers::StackConfig stack;
  stack.hidden_size = config.hidden_size;
  stack.vocab_size = config.vocab_size;
  stack.embedding_scale = std::sqrt(static_cast<float>(config.hidden_size));
  stack.final_norm = norm;
  stack.blocks.resize(config.num_hidden_layers);
  for (size_t i = 0; i < stack.blocks.size(); ++i) {
    auto& block = stack.blocks[i];
    block.norm = norm;
    block.residual = layers::ResidualStyle::kOutputNorm;
    block.mixer_out_norm = norm;
    block.feed_forward_out_norm = norm;
    layers::AttentionConfig layer = attention;
    const bool global =
        config.sliding_window_pattern > 0 && (i + 1) % config.sliding_window_pattern == 0;
    if (!global) {
      layer.rotary.theta = static_cast<float>(config.rope_local_base_freq);
    }
    layer.sliding_window = global || config.sliding_window <= 0 ? 0 : config.sliding_window;
    block.mixer = layer;
    block.feed_forward =
        layers::GatedMlpConfig{config.intermediate_size, config.activation};
  }
  return stack;
}

}  // namespace inferx::models::gemma3
