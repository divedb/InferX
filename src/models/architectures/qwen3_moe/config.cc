#include "inferx/models/architectures/qwen3_moe/config.h"

namespace inferx::models::qwen3_moe {

StatusOr<Qwen3MoeConfig> Qwen3MoeConfig::ParseBase(const ConfigView& view) {
  Qwen3MoeConfig config;
  INFERX_ASSIGN_OR_RETURN(auto dims, ModelDims::Parse(view));
  static_cast<ModelDims&>(config) = std::move(dims);
  INFERX_ASSIGN_OR_RETURN(
      config.rope, RopeConfig::Parse(view, config.head_dim, config.max_position_embeddings));
  INFERX_ASSIGN_OR_RETURN(config.num_experts, view.RequireInt("num_experts"));
  INFERX_ASSIGN_OR_RETURN(config.num_experts_per_tok, view.RequireInt("num_experts_per_tok"));
  INFERX_ASSIGN_OR_RETURN(config.moe_intermediate_size,
                          view.RequireInt("moe_intermediate_size"));
  INFERX_ASSIGN_OR_RETURN(config.norm_topk_prob, view.Bool("norm_topk_prob", true));
  INFERX_ASSIGN_OR_RETURN(config.shared_expert_intermediate_size,
                          view.Int("shared_expert_intermediate_size", 0));
  INFERX_ASSIGN_OR_RETURN(config.decoder_sparse_step, view.Int("decoder_sparse_step", 1));
  INFERX_ASSIGN_OR_RETURN(config.mlp_only_layers, view.IntList("mlp_only_layers"));
  return config;
}

StatusOr<Qwen3MoeConfig> Qwen3MoeConfig::Parse(const ConfigView& view) {
  INFERX_ASSIGN_OR_RETURN(auto config, ParseBase(view));
  INFERX_RETURN_IF_ERROR(config.Validate());
  return config;
}

Status Qwen3MoeConfig::Validate() const {
  INFERX_RETURN_IF_ERROR(ModelDims::Validate());
  INFERX_RETURN_IF_ERROR(rope.Validate(head_dim));
  if (num_experts <= 0 || num_experts_per_tok <= 0 ||
      num_experts_per_tok > num_experts || moe_intermediate_size <= 0 ||
      shared_expert_intermediate_size < 0) {
    return InvalidArgumentError("invalid Qwen3-MoE expert configuration");
  }
  if (decoder_sparse_step <= 0) {
    return InvalidArgumentError("decoder_sparse_step must be positive");
  }
  for (const int64_t layer : mlp_only_layers) {
    if (layer < 0 || layer >= num_hidden_layers) {
      return InvalidArgumentError("mlp_only_layers index is out of range");
    }
  }
  return OkStatus();
}

}  // namespace inferx::models::qwen3_moe
