#include "inferx/models/architectures/deepseek_v3/config.h"

namespace inferx::models::deepseek_v3 {

StatusOr<DeepseekV3Config> DeepseekV3Config::Parse(const ConfigView& view) {
  DeepseekV3Config config;
  INFERX_ASSIGN_OR_RETURN(auto dims, ModelDims::Parse(view));
  static_cast<ModelDims&>(config) = std::move(dims);

  INFERX_ASSIGN_OR_RETURN(config.q_lora_rank, view.RequireInt("q_lora_rank"));
  INFERX_ASSIGN_OR_RETURN(config.kv_lora_rank, view.RequireInt("kv_lora_rank"));
  INFERX_ASSIGN_OR_RETURN(config.qk_nope_head_dim, view.RequireInt("qk_nope_head_dim"));
  INFERX_ASSIGN_OR_RETURN(config.qk_rope_head_dim, view.RequireInt("qk_rope_head_dim"));
  INFERX_ASSIGN_OR_RETURN(config.v_head_dim, view.RequireInt("v_head_dim"));
  // The rotary window is the per-head rope width, not the attention head
  // dimension of a GQA model.
  INFERX_ASSIGN_OR_RETURN(
      config.rope, RopeConfig::Parse(view, config.qk_rope_head_dim,
                                     config.max_position_embeddings));
  config.rope.dim = config.qk_rope_head_dim;

  INFERX_ASSIGN_OR_RETURN(config.n_routed_experts, view.RequireInt("n_routed_experts"));
  INFERX_ASSIGN_OR_RETURN(config.num_experts_per_tok, view.RequireInt("num_experts_per_tok"));
  INFERX_ASSIGN_OR_RETURN(config.moe_intermediate_size,
                          view.RequireInt("moe_intermediate_size"));
  INFERX_ASSIGN_OR_RETURN(config.norm_topk_prob, view.Bool("norm_topk_prob", true));
  INFERX_ASSIGN_OR_RETURN(config.routed_scaling_factor,
                          view.Float("routed_scaling_factor", 1.0));
  INFERX_ASSIGN_OR_RETURN(config.n_group, view.Int("n_group", 1));
  INFERX_ASSIGN_OR_RETURN(config.topk_group, view.Int("topk_group", 0));
  if (config.topk_group <= 0) config.topk_group = config.n_group;
  INFERX_ASSIGN_OR_RETURN(config.first_k_dense_replace,
                          view.Int("first_k_dense_replace", 0));

  INFERX_RETURN_IF_ERROR(config.Validate());
  return config;
}

Status DeepseekV3Config::Validate() const {
  INFERX_RETURN_IF_ERROR(ModelDims::Validate());
  INFERX_RETURN_IF_ERROR(rope.Validate(qk_rope_head_dim));
  if (q_lora_rank <= 0 || kv_lora_rank <= 0 || qk_nope_head_dim <= 0 ||
      qk_rope_head_dim <= 0 || v_head_dim <= 0 || v_head_dim > head_dim()) {
    return InvalidArgumentError("invalid DeepSeek multi-latent attention geometry");
  }
  if (n_routed_experts <= 0 || num_experts_per_tok <= 0 ||
      num_experts_per_tok > n_routed_experts || moe_intermediate_size <= 0 ||
      n_group <= 0 || topk_group <= 0 || first_k_dense_replace < 0 ||
      first_k_dense_replace > num_hidden_layers) {
    return InvalidArgumentError("invalid DeepSeek expert configuration");
  }
  if (n_group > 1) {
    if (n_routed_experts % n_group != 0 ||
        topk_group * (n_routed_experts / n_group) < num_experts_per_tok) {
      return InvalidArgumentError("grouped routing cannot select enough experts");
    }
  }
  return OkStatus();
}

}  // namespace inferx::models::deepseek_v3
