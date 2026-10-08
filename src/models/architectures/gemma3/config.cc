#include "inferx/models/architectures/gemma3/config.h"

namespace inferx::models::gemma3 {

StatusOr<Gemma3Config> Gemma3Config::Parse(const ConfigView& view) {
  Gemma3Config config;
  INFERX_ASSIGN_OR_RETURN(auto dims, ModelDims::Parse(view));
  static_cast<ModelDims&>(config) = std::move(dims);
  INFERX_ASSIGN_OR_RETURN(
      config.rope, RopeConfig::Parse(view, config.head_dim, config.max_position_embeddings));
  INFERX_ASSIGN_OR_RETURN(config.rope_local_base_freq,
                          view.Float("rope_local_base_freq", 10000.0));
  INFERX_ASSIGN_OR_RETURN(config.query_pre_attn_scalar,
                          view.Float("query_pre_attn_scalar", 0.0));
  INFERX_ASSIGN_OR_RETURN(config.sliding_window, view.Int("sliding_window", 0));
  INFERX_ASSIGN_OR_RETURN(config.sliding_window_pattern, view.Int("sliding_window_pattern", 6));
  INFERX_RETURN_IF_ERROR(config.Validate());
  return config;
}

Status Gemma3Config::Validate() const {
  INFERX_RETURN_IF_ERROR(ModelDims::Validate());
  INFERX_RETURN_IF_ERROR(rope.Validate(head_dim));
  if (rope_local_base_freq <= 0) {
    return InvalidArgumentError("rope_local_base_freq must be positive");
  }
  if (sliding_window < 0 || sliding_window_pattern < 0) {
    return InvalidArgumentError("sliding_window and pattern must not be negative");
  }
  return OkStatus();
}

}  // namespace inferx::models::gemma3
