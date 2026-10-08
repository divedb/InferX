#include "inferx/models/architectures/mistral/config.h"

namespace inferx::models::mistral {

StatusOr<MistralConfig> MistralConfig::Parse(const ConfigView& view) {
  MistralConfig config;
  INFERX_ASSIGN_OR_RETURN(auto dims, ModelDims::Parse(view));
  static_cast<ModelDims&>(config) = std::move(dims);
  INFERX_ASSIGN_OR_RETURN(
      config.rope, RopeConfig::Parse(view, config.head_dim, config.max_position_embeddings));
  INFERX_ASSIGN_OR_RETURN(config.sliding_window, view.Int("sliding_window", 0));
  INFERX_ASSIGN_OR_RETURN(config.use_sliding_window, view.Bool("use_sliding_window", false));
  INFERX_RETURN_IF_ERROR(config.Validate());
  return config;
}

Status MistralConfig::Validate() const {
  INFERX_RETURN_IF_ERROR(ModelDims::Validate());
  INFERX_RETURN_IF_ERROR(rope.Validate(head_dim));
  if (sliding_window < 0) return InvalidArgumentError("sliding_window must not be negative");
  return OkStatus();
}

}  // namespace inferx::models::mistral
