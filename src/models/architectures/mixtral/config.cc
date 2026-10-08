#include "inferx/models/architectures/mixtral/config.h"

namespace inferx::models::mixtral {

StatusOr<MixtralConfig> MixtralConfig::Parse(const ConfigView& view) {
  MixtralConfig config;
  INFERX_ASSIGN_OR_RETURN(auto dims, ModelDims::Parse(view));
  static_cast<ModelDims&>(config) = std::move(dims);
  INFERX_ASSIGN_OR_RETURN(
      config.rope, RopeConfig::Parse(view, config.head_dim, config.max_position_embeddings));
  INFERX_ASSIGN_OR_RETURN(config.num_local_experts, view.RequireInt("num_local_experts"));
  INFERX_ASSIGN_OR_RETURN(config.num_experts_per_tok, view.RequireInt("num_experts_per_tok"));
  INFERX_ASSIGN_OR_RETURN(config.norm_topk_prob, view.Bool("norm_topk_prob", true));
  INFERX_ASSIGN_OR_RETURN(config.sliding_window, view.Int("sliding_window", 0));
  INFERX_ASSIGN_OR_RETURN(config.use_sliding_window, view.Bool("use_sliding_window", true));
  INFERX_RETURN_IF_ERROR(config.Validate());
  return config;
}

Status MixtralConfig::Validate() const {
  INFERX_RETURN_IF_ERROR(ModelDims::Validate());
  INFERX_RETURN_IF_ERROR(rope.Validate(head_dim));
  if (num_local_experts <= 0 || num_experts_per_tok <= 0 ||
      num_experts_per_tok > num_local_experts) {
    return InvalidArgumentError("invalid Mixtral expert counts");
  }
  return OkStatus();
}

}  // namespace inferx::models::mixtral
