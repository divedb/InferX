/// \file
/// \brief Mistral dense configuration: optional sliding-window attention
/// over an otherwise canonical pre-norm decoder.

#ifndef INFERX_MODELS_ARCHITECTURES_MISTRAL_CONFIG_H_
#define INFERX_MODELS_ARCHITECTURES_MISTRAL_CONFIG_H_

#include <cstdint>

#include "inferx/core/status.h"
#include "inferx/models/configs/dims.h"
#include "inferx/models/configs/rope.h"
#include "inferx/util/config_view.h"

namespace inferx::models::mistral {

struct MistralConfig : ModelDims {
  RopeConfig rope;
  /// Attention window in tokens; 0 disables windowing.
  int64_t sliding_window = 0;
  /// Whether the checkpoint enables the sliding window at all.
  bool use_sliding_window = false;

  static StatusOr<MistralConfig> Parse(const ConfigView& view);
  Status Validate() const;
};

}  // namespace inferx::models::mistral

#endif  // INFERX_MODELS_ARCHITECTURES_MISTRAL_CONFIG_H_
