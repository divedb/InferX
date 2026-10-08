#include "inferx/models/architectures/qwen2/config.h"

namespace inferx::models::qwen2 {

StatusOr<Qwen2Config> Qwen2Config::Parse(const ConfigView& view) {
  Qwen2Config config;
  INFERX_ASSIGN_OR_RETURN(auto dims, ModelDims::Parse(view));
  static_cast<ModelDims&>(config) = std::move(dims);
  INFERX_ASSIGN_OR_RETURN(
      config.rope, RopeConfig::Parse(view, config.head_dim, config.max_position_embeddings));
  INFERX_RETURN_IF_ERROR(config.Validate());
  return config;
}

Status Qwen2Config::Validate() const {
  INFERX_RETURN_IF_ERROR(ModelDims::Validate());
  return rope.Validate(head_dim);
}

}  // namespace inferx::models::qwen2
