/// \file
/// \brief Qwen3-Next configuration: a recurrent-hybrid schedule of
/// Gated DeltaNet linear-attention layers and sigmoid-gated full
/// attention layers, each every sparse step routing through experts with
/// a gated shared expert. Separate type from Qwen3MoeConfig.

#ifndef INFERX_MODELS_ARCHITECTURES_QWEN3_NEXT_CONFIG_H_
#define INFERX_MODELS_ARCHITECTURES_QWEN3_NEXT_CONFIG_H_

#include <cstdint>
#include <string>
#include <vector>

#include "inferx/core/status.h"
#include "inferx/models/architectures/qwen3_moe/config.h"
#include "inferx/util/config_view.h"

namespace inferx::models::qwen3_next {

/// Layer mixer selection: "full_attention" or "linear_attention".
enum class LayerType { kFullAttention, kLinearAttention };

struct Qwen3NextConfig : qwen3_moe::Qwen3MoeConfig {
  /// Gated DeltaNet geometry (linear-attention layers).
  int64_t linear_num_key_heads = 0;
  int64_t linear_num_value_heads = 0;
  int64_t linear_key_head_dim = 0;
  int64_t linear_value_head_dim = 0;
  int64_t linear_conv_kernel_dim = 0;
  /// Default schedule: every interval-th layer attends fully.
  int64_t full_attention_interval = 4;
  /// Explicit per-layer schedule; empty selects the interval default.
  std::vector<LayerType> layer_types;

  static StatusOr<Qwen3NextConfig> Parse(const ConfigView& view);
  Status Validate() const;
};

}  // namespace inferx::models::qwen3_next

#endif  // INFERX_MODELS_ARCHITECTURES_QWEN3_NEXT_CONFIG_H_
