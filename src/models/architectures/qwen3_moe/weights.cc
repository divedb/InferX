/// \file
/// \brief Qwen3-MoE checkpoint mapping and composition: Qwen3 attention
/// (Q/K RMSNorm + RoPE) with softmax-topk experts every sparse step,
/// skipped for the configured dense layers.

#include <algorithm>

#include "inferx/models/architectures/qwen3_moe/model.h"

#include "inferx/models/layers/decoder.h"

namespace inferx::models::qwen3_moe {

const WeightNames& CheckpointNames() {
  static const WeightNames names{};
  return names;
}

const WeightLayout& CheckpointLayout() {
  static const WeightLayout layout{};
  return layout;
}

StatusOr<layers::StackConfig> BuildStackConfig(const Qwen3MoeConfig& config) {
  layers::AttentionConfig attention;
  attention.query_heads = config.num_attention_heads;
  attention.kv_heads = config.num_key_value_heads;
  attention.head_dim = config.head_dim;
  attention.qk_norm = true;
  attention.qkv_bias = config.attention_bias;
  attention.output_bias = config.attention_bias;
  attention.rotary = config.rope.Params();

  layers::MoeConfig experts;
  experts.num_experts = config.num_experts;
  experts.experts_per_token = config.num_experts_per_tok;
  experts.intermediate_size = config.moe_intermediate_size;
  experts.activation = config.activation;
  experts.oai_limit = config.swiglu_limit;
  experts.routing.normalize = config.norm_topk_prob;
  experts.shared_intermediate_size = config.shared_expert_intermediate_size;

  layers::StackConfig stack;
  stack.hidden_size = config.hidden_size;
  stack.vocab_size = config.vocab_size;
  stack.final_norm = layers::NormConfig{static_cast<float>(config.rms_norm_eps)};
  stack.blocks.resize(config.num_hidden_layers);
  for (size_t i = 0; i < stack.blocks.size(); ++i) {
    auto& block = stack.blocks[i];
    block.norm = stack.final_norm;
    block.mixer = attention;
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

}  // namespace inferx::models::qwen3_moe
