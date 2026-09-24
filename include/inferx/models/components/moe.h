/// \file
/// \brief Top-k routed mixture of experts.

#ifndef INFERX_MODELS_COMPONENTS_MOE_H_
#define INFERX_MODELS_COMPONENTS_MOE_H_

#include <cstdint>
#include <optional>
#include <vector>

#include "inferx/core/tensor.h"
#include "inferx/models/components/mlp.h"
#include "inferx/ops/execution_context.h"

namespace inferx::components {

/// \brief Top-k routed mixture of SwiGLU experts.
struct MoeConfig {
  int64_t num_experts = 0;
  int64_t experts_per_token = 0;
  int64_t intermediate_size = 0;         ///< Per-expert gate/up width.
  bool normalize_routing_weights = true;
  int64_t shared_intermediate_size = 0;  ///< 0 disables the shared expert.
  bool gate_shared_expert = false;
};

/// \brief Router, routed experts, and optional shared expert.
struct MoeWeights {
  Tensor router;                       ///< [num_experts, hidden]
  std::vector<SwiGluWeights> experts;  ///< One per routed expert.
  std::optional<SwiGluWeights> shared_expert;  ///< Absent when disabled.
  std::optional<Tensor> shared_expert_gate;    ///< [1, hidden]; present only when gated.
};

/// \brief Runs the routed expert feed-forward into `mixed_out`.
///
/// The single home for expert execution; until it is implemented,
/// ValidateExecutable() rejects MoE models at build time.
Status RunMoe(const MoeConfig& config, const MoeWeights& weights, const Tensor& normed,
              ops::ExecutionContext& ctx, Tensor& mixed_out);

}  // namespace inferx::components

#endif  // INFERX_MODELS_COMPONENTS_MOE_H_
