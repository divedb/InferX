#include "inferx/models/architectures/gpt_oss/config.h"

namespace inferx::models::gpt_oss {

StatusOr<GptOssConfig> GptOssConfig::Parse(const ConfigView& view) {
  GptOssConfig config;
  INFERX_ASSIGN_OR_RETURN(auto dims, ModelDims::Parse(view));
  static_cast<ModelDims&>(config) = std::move(dims);
  INFERX_ASSIGN_OR_RETURN(
      config.rope, RopeConfig::Parse(view, config.head_dim, config.max_position_embeddings));
  INFERX_ASSIGN_OR_RETURN(config.num_local_experts, view.RequireInt("num_local_experts"));
  INFERX_ASSIGN_OR_RETURN(config.num_experts_per_tok, view.RequireInt("num_experts_per_tok"));
  // gpt-oss's `intermediate_size` is the per-expert width.
  INFERX_ASSIGN_OR_RETURN(config.expert_intermediate_size,
                          view.RequireInt("intermediate_size"));
  INFERX_ASSIGN_OR_RETURN(config.sliding_window, view.Int("sliding_window", 0));

  INFERX_ASSIGN_OR_RETURN(config.layer_types, view.StrList("layer_types"));
  for (const auto& type : config.layer_types) {
    if (type != "sliding_attention" && type != "full_attention") {
      return UnimplementedError("unsupported gpt-oss layer type: ", type);
    }
  }
  if (!config.layer_types.empty() &&
      config.layer_types.size() != static_cast<size_t>(config.num_hidden_layers)) {
    return InvalidArgumentError("layer_types must have one entry per decoder layer");
  }

  INFERX_RETURN_IF_ERROR(config.Validate());
  return config;
}

Status GptOssConfig::Validate() const {
  INFERX_RETURN_IF_ERROR(ModelDims::Validate());
  INFERX_RETURN_IF_ERROR(rope.Validate(head_dim));
  if (num_local_experts <= 0 || num_experts_per_tok <= 0 ||
      num_experts_per_tok > num_local_experts || expert_intermediate_size <= 0 ||
      sliding_window < 0) {
    return InvalidArgumentError("invalid gpt-oss expert configuration");
  }
  return OkStatus();
}

}  // namespace inferx::models::gpt_oss
