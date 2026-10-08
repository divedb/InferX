#include "inferx/models/architectures/qwen3/config.h"

namespace inferx::models::qwen3 {

StatusOr<Qwen3Config> Qwen3Config::Parse(const ConfigView& view) {
  // Heterogeneous layer schedules change execution; they belong to the
  // Qwen3-Next architecture.
  INFERX_RETURN_IF_ERROR(view.RejectIfPresent(
      "layer_types", "heterogeneous layer schedules (see qwen3_next)"));
  Qwen3Config config;
  INFERX_ASSIGN_OR_RETURN(auto dims, ModelDims::Parse(view));
  static_cast<ModelDims&>(config) = std::move(dims);
  INFERX_ASSIGN_OR_RETURN(
      config.rope, RopeConfig::Parse(view, config.head_dim, config.max_position_embeddings));
  INFERX_RETURN_IF_ERROR(config.Validate());
  return config;
}

Status Qwen3Config::Validate() const {
  INFERX_RETURN_IF_ERROR(ModelDims::Validate());
  return rope.Validate(head_dim);
}

}  // namespace inferx::models::qwen3
