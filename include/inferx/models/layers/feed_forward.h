/// \file
/// \brief Feed-forward components: gated dense MLPs and routed experts.

#ifndef INFERX_MODELS_LAYERS_FEED_FORWARD_H_
#define INFERX_MODELS_LAYERS_FEED_FORWARD_H_

#include <cstdint>
#include <optional>
#include <vector>

#include "inferx/core/tensor.h"
#include "inferx/models/layers/mixer.h"

namespace inferx::layers {

/// \brief Bias-free SwiGLU feed-forward.
struct SwiGluConfig {
  int64_t intermediate_size = 0;  ///< Gate/up width.
};

/// \brief Top-k routed mixture of SwiGLU experts.
struct MoeConfig {
  int64_t num_experts = 0;
  int64_t experts_per_token = 0;
  int64_t intermediate_size = 0;         ///< Per-expert gate/up width.
  bool normalize_routing_weights = true;
  int64_t shared_intermediate_size = 0;  ///< 0 disables the shared expert.
  bool gate_shared_expert = false;
};

/// \brief SwiGLU projections for one expert or a dense layer.
struct SwiGluWeights {
  std::optional<Tensor> packed_gate_up;  ///< Concatenated gate/up rows, when packed.
  LinearWeights gate;  ///< [intermediate, hidden]
  LinearWeights up;    ///< [intermediate, hidden]
  LinearWeights down;  ///< [hidden, intermediate]
};

/// \brief Router, routed experts, and optional shared expert.
struct MoeWeights {
  Tensor router;                       ///< [num_experts, hidden]
  std::vector<SwiGluWeights> experts;  ///< One per routed expert.
  std::optional<SwiGluWeights> shared_expert;  ///< Absent when disabled.
  std::optional<Tensor> shared_expert_gate;    ///< [1, hidden]; present only when gated.
};

}  // namespace inferx::layers

#endif  // INFERX_MODELS_LAYERS_FEED_FORWARD_H_
