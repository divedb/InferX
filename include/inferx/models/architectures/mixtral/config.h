/// \file
/// \brief Mixtral configuration: softmax top-k routed experts with the
/// model intermediate size as the per-expert width, plus optional
/// sliding-window attention.

#ifndef INFERX_MODELS_ARCHITECTURES_MIXTRAL_CONFIG_H_
#define INFERX_MODELS_ARCHITECTURES_MIXTRAL_CONFIG_H_

#include <cstdint>

#include "inferx/core/status.h"
#include "inferx/models/configs/dims.h"
#include "inferx/models/configs/rope.h"
#include "inferx/util/config_view.h"

namespace inferx::models::mixtral {

struct MixtralConfig : ModelDims {
  RopeConfig rope;
  int64_t num_local_experts = 0;
  int64_t num_experts_per_tok = 0;
  /// Renormalize selected expert weights to sum to one.
  bool norm_topk_prob = true;
  /// Attention window in tokens; 0 disables windowing.
  int64_t sliding_window = 0;
  bool use_sliding_window = true;

  static StatusOr<MixtralConfig> Parse(const ConfigView& view);
  Status Validate() const;
};

}  // namespace inferx::models::mixtral

#endif  // INFERX_MODELS_ARCHITECTURES_MIXTRAL_CONFIG_H_
