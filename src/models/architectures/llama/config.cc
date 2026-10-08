#include "inferx/models/architectures/llama/config.h"

namespace inferx::models::llama {

StatusOr<LlamaConfig> LlamaConfig::Parse(const ConfigView& view) {
  LlamaConfig config;
  INFERX_ASSIGN_OR_RETURN(auto dims, ModelDims::Parse(view));
  static_cast<ModelDims&>(config) = std::move(dims);
  INFERX_ASSIGN_OR_RETURN(
      config.rope, RopeConfig::Parse(view, config.head_dim, config.max_position_embeddings));
  INFERX_RETURN_IF_ERROR(config.Validate());
  return config;
}

Status LlamaConfig::Validate() const {
  INFERX_RETURN_IF_ERROR(ModelDims::Validate());
  return rope.Validate(head_dim);
}

}  // namespace inferx::models::llama
