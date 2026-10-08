/// \file
/// \brief gpt-oss configuration: attention with per-head sinks and
/// biased projections, SwiGLU-oai activations, and every expert fused
/// into MXFP4-quantized tensors with interleaved rows.

#ifndef INFERX_MODELS_ARCHITECTURES_GPT_OSS_CONFIG_H_
#define INFERX_MODELS_ARCHITECTURES_GPT_OSS_CONFIG_H_

#include <cstdint>
#include <string>
#include <vector>

#include "inferx/core/status.h"
#include "inferx/models/configs/dims.h"
#include "inferx/models/configs/rope.h"
#include "inferx/util/config_view.h"

namespace inferx::models::gpt_oss {

struct GptOssConfig : ModelDims {
  RopeConfig rope;
  int64_t num_local_experts = 0;
  int64_t num_experts_per_tok = 0;
  /// Per-expert intermediate width (`intermediate_size` in config.json).
  int64_t expert_intermediate_size = 0;
  /// Attention window in tokens for sliding layers; 0 disables windowing.
  int64_t sliding_window = 0;
  /// Explicit schedule: "sliding_attention" or "full_attention" per layer.
  std::vector<std::string> layer_types;

  static StatusOr<GptOssConfig> Parse(const ConfigView& view);
  Status Validate() const;
};

}  // namespace inferx::models::gpt_oss

#endif  // INFERX_MODELS_ARCHITECTURES_GPT_OSS_CONFIG_H_
