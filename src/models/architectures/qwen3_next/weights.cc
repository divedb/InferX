/// \file
/// \brief Qwen3-Next checkpoint mapping and composition. The layer
/// schedule interleaves sigmoid-gated full attention with Gated DeltaNet
/// linear attention; norms scale by (1 + w); the shared expert output is
/// sigmoid-gated.

#include <algorithm>

#include "inferx/models/architectures/qwen3_next/model.h"

#include "inferx/models/layers/decoder.h"

namespace inferx::models::qwen3_next {

const WeightNames& CheckpointNames() {
  static const WeightNames names{};
  return names;
}

const WeightLayout& CheckpointLayout() {
  static const WeightLayout layout{};
  return layout;
}

StatusOr<layers::StackConfig> BuildStackConfig(const Qwen3NextConfig& config) {
  layers::AttentionConfig attention;
  attention.query_heads = config.num_attention_heads;
  attention.kv_heads = config.num_key_value_heads;
  attention.head_dim = config.head_dim;
  attention.qk_norm = true;
  attention.qk_norm_plus_one = true;
  attention.qkv_bias = config.attention_bias;
  attention.output_bias = config.attention_bias;
  attention.output_gate = layers::OutputGate::kSigmoid;
  attention.rotary = config.rope.Params();

  const layers::GatedDeltaNetConfig linear{config.linear_num_key_heads,
                                           config.linear_num_value_heads,
                                           config.linear_key_head_dim,
                                           config.linear_value_head_dim,
                                           config.linear_conv_kernel_dim};

  layers::MoeConfig experts;
  experts.num_experts = config.num_experts;
  experts.experts_per_token = config.num_experts_per_tok;
  experts.intermediate_size = config.moe_intermediate_size;
  experts.activation = config.activation;
  experts.oai_limit = config.swiglu_limit;
  experts.routing.normalize = config.norm_topk_prob;
  experts.shared_intermediate_size = config.shared_expert_intermediate_size;
  experts.gate_shared_expert = config.shared_expert_intermediate_size > 0;

  const layers::NormConfig norm{static_cast<float>(config.rms_norm_eps),
                                 /*plus_one=*/true};

  layers::StackConfig stack;
  stack.hidden_size = config.hidden_size;
  stack.vocab_size = config.vocab_size;
  stack.final_norm = norm;
  stack.blocks.resize(config.num_hidden_layers);
  for (size_t i = 0; i < stack.blocks.size(); ++i) {
    auto& block = stack.blocks[i];
    block.norm = norm;
    const bool full =
        config.layer_types.empty()
            ? (i + 1) % config.full_attention_interval == 0
            : config.layer_types[i] == LayerType::kFullAttention;
    block.mixer = full ? layers::DecoderLayerConfig::Mixer{attention}
                       : layers::DecoderLayerConfig::Mixer{linear};
    const bool routed = (i + 1) % config.decoder_sparse_step == 0 &&
                        std::find(config.mlp_only_layers.begin(), config.mlp_only_layers.end(),
                                  static_cast<int64_t>(i)) == config.mlp_only_layers.end();
    block.feed_forward =
        routed ? layers::DecoderLayerConfig::FeedForward{experts}
               : layers::DecoderLayerConfig::FeedForward{layers::GatedMlpConfig{
                     config.intermediate_size, config.activation}};
  }
  return stack;
}

}  // namespace inferx::models::qwen3_next
