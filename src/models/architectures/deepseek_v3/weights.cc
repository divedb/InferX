/// \file
/// \brief DeepSeek-V3 checkpoint mapping and composition: MLA mixers
/// (rope over the per-head rope width) and grouped-routing experts with a
/// correction bias and a dense shared expert; a dense prefix keeps the
/// plain gated MLP.

#include "inferx/models/architectures/deepseek_v3/model.h"

#include "inferx/models/layers/decoder.h"
#include "inferx/ops/moe.h"

namespace inferx::models::deepseek_v3 {

const WeightNames& CheckpointNames() {
  static const WeightNames names{};
  return names;
}

const WeightLayout& CheckpointLayout() {
  static const WeightLayout layout{};
  return layout;
}

StatusOr<layers::StackConfig> BuildStackConfig(const DeepseekV3Config& config) {
  layers::MlaConfig mla;
  mla.query_heads = config.num_attention_heads;
  mla.q_lora_rank = config.q_lora_rank;
  mla.kv_lora_rank = config.kv_lora_rank;
  mla.qk_nope_head_dim = config.qk_nope_head_dim;
  mla.qk_rope_head_dim = config.qk_rope_head_dim;
  mla.v_head_dim = config.v_head_dim;
  mla.rotary = config.rope.Params();

  layers::MoeConfig experts;
  experts.num_experts = config.n_routed_experts;
  experts.experts_per_token = config.num_experts_per_tok;
  experts.intermediate_size = config.moe_intermediate_size;
  experts.activation = config.activation;
  experts.oai_limit = config.swiglu_limit;
  experts.routing.scoring = ops::RouterScoring::kSigmoidGroupTopk;
  experts.routing.normalize = config.norm_topk_prob;
  experts.routing.routing_scale = static_cast<float>(config.routed_scaling_factor);
  experts.routing.group_count = config.n_group;
  experts.routing.group_topk = config.topk_group;
  experts.has_correction_bias = true;

  layers::StackConfig stack;
  stack.hidden_size = config.hidden_size;
  stack.vocab_size = config.vocab_size;
  stack.final_norm = layers::NormConfig{static_cast<float>(config.rms_norm_eps)};
  stack.blocks.resize(config.num_hidden_layers);
  for (size_t i = 0; i < stack.blocks.size(); ++i) {
    auto& block = stack.blocks[i];
    block.norm = stack.final_norm;
    block.mixer = mla;
    block.feed_forward =
        static_cast<int64_t>(i) >= config.first_k_dense_replace
            ? layers::DecoderLayerConfig::FeedForward{experts}
            : layers::DecoderLayerConfig::FeedForward{layers::GatedMlpConfig{
                  config.intermediate_size, config.activation}};
  }
  return stack;
}

}  // namespace inferx::models::deepseek_v3
