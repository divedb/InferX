/// \file
/// \brief Gemma3 text configuration: sandwich (output-side) norms,
/// plus-one RMSNorm, GeGLU, scaled embeddings, alternating local/global
/// attention with local rope frequencies, and a windowed local pattern.

#ifndef INFERX_MODELS_ARCHITECTURES_GEMMA3_CONFIG_H_
#define INFERX_MODELS_ARCHITECTURES_GEMMA3_CONFIG_H_

#include <cstdint>

#include "inferx/core/status.h"
#include "inferx/models/configs/dims.h"
#include "inferx/models/configs/rope.h"
#include "inferx/util/config_view.h"

namespace inferx::models::gemma3 {

struct Gemma3Config : ModelDims {
  RopeConfig rope;  ///< Global-layer rope; local layers override theta.
  /// Base frequency of local (windowed) attention layers.
  double rope_local_base_freq = 10000.0;
  /// Attention logit denominator from query_pre_attn_scalar; 0 derives
  /// from head_dim.
  double query_pre_attn_scalar = 0.0;
  /// Attention window in tokens; 0 disables windowing.
  int64_t sliding_window = 0;
  /// Every pattern-th layer attends globally (0 disables the pattern).
  int64_t sliding_window_pattern = 6;

  static StatusOr<Gemma3Config> Parse(const ConfigView& view);
  Status Validate() const;
};

}  // namespace inferx::models::gemma3

#endif  // INFERX_MODELS_ARCHITECTURES_GEMMA3_CONFIG_H_
