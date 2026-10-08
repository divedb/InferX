#include "inferx/models/architectures/qwen3_next/config.h"

#include <algorithm>

namespace inferx::models::qwen3_next {

StatusOr<Qwen3NextConfig> Qwen3NextConfig::Parse(const ConfigView& view) {
  Qwen3NextConfig config;
  // Shared Qwen3-MoE core: dimensions, rope, expert fields.
  INFERX_ASSIGN_OR_RETURN(auto base, qwen3_moe::Qwen3MoeConfig::ParseBase(view));
  static_cast<qwen3_moe::Qwen3MoeConfig&>(config) = std::move(base);

  INFERX_ASSIGN_OR_RETURN(config.full_attention_interval,
                          view.Int("full_attention_interval", 4));
  INFERX_ASSIGN_OR_RETURN(config.linear_num_key_heads, view.Int("linear_num_key_heads", 0));
  INFERX_ASSIGN_OR_RETURN(config.linear_num_value_heads,
                          view.Int("linear_num_value_heads", 0));
  INFERX_ASSIGN_OR_RETURN(config.linear_key_head_dim, view.Int("linear_key_head_dim", 0));
  INFERX_ASSIGN_OR_RETURN(config.linear_value_head_dim, view.Int("linear_value_head_dim", 0));
  INFERX_ASSIGN_OR_RETURN(config.linear_conv_kernel_dim,
                          view.Int("linear_conv_kernel_dim", 0));

  INFERX_ASSIGN_OR_RETURN(const auto raw_types, view.StrList("layer_types"));
  if (!raw_types.empty()) {
    if (raw_types.size() != static_cast<size_t>(config.num_hidden_layers)) {
      return InvalidArgumentError("layer_types must have one entry per decoder layer");
    }
    config.layer_types.reserve(raw_types.size());
    for (const auto& raw : raw_types) {
      if (raw == "full_attention") {
        config.layer_types.push_back(LayerType::kFullAttention);
      } else if (raw == "linear_attention") {
        config.layer_types.push_back(LayerType::kLinearAttention);
      } else {
        return UnimplementedError("unsupported Qwen3-Next layer type: ", raw);
      }
    }
  }

  INFERX_RETURN_IF_ERROR(config.Validate());
  return config;
}

Status Qwen3NextConfig::Validate() const {
  INFERX_RETURN_IF_ERROR(qwen3_moe::Qwen3MoeConfig::Validate());
  if (full_attention_interval <= 0) {
    return InvalidArgumentError("full_attention_interval must be positive");
  }
  // Default schedule: full attention when (i + 1) % interval == 0, so any
  // interval above 1 schedules at least one linear layer.
  const bool any_linear =
      layer_types.empty()
          ? full_attention_interval > 1
          : std::any_of(layer_types.begin(), layer_types.end(),
                        [](LayerType t) { return t == LayerType::kLinearAttention; });
  if (any_linear &&
      (linear_num_key_heads <= 0 || linear_num_value_heads <= 0 || linear_key_head_dim <= 0 ||
       linear_value_head_dim <= 0 || linear_conv_kernel_dim <= 0)) {
    return InvalidArgumentError(
        "linear attention layers require the linear_* head and kernel dimensions");
  }
  if (!layer_types.empty() &&
      layer_types.size() != static_cast<size_t>(num_hidden_layers)) {
    return InvalidArgumentError("layer_types must have one entry per decoder layer");
  }
  return OkStatus();
}

}  // namespace inferx::models::qwen3_next
