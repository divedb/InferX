#ifndef INFERX_MODELS_QWEN3_MOE_QWEN3_MOE_CONFIG_H_
#define INFERX_MODELS_QWEN3_MOE_QWEN3_MOE_CONFIG_H_

#include <cstdint>

#include "inferx/core/status.h"
#include "inferx/models/qwen3/qwen3_config.h"
#include "inferx/ops/moe.h"
#include "inferx/util/hf_config.h"

namespace inferx {
namespace models {
namespace qwen3_moe {

/// \brief The Qwen3-MoE architecture's checked `config.json` fields.
///
/// Composes the dense Qwen3 geometry (attention is identical) and adds the
/// expert configuration. Routing is softmax-top-k with per-checkpoint
/// renormalization; the policy object is derived once here.
struct Qwen3MoeConfig {
  /// Shared dense geometry: heads, rope, eps, vocabulary.
  qwen3::Qwen3Config base;

  int64_t num_experts = 0;             ///< Total experts E.
  int64_t num_experts_per_tok = 0;     ///< Experts routed per token.
  bool norm_topk_prob = true;          ///< Renormalize the top-k weights.
  int64_t shared_expert_intermediate = 0;  ///< 0: no shared expert.
  bool shared_expert_gate = false;     ///< Sigmoid gate on the shared expert.

  /// \brief Parses and validates the checkpoint's text config.
  static StatusOr<Qwen3MoeConfig> FromHf(const HfConfig& config);

  /// \brief The routing policy this family runs.
  ops::RoutingConfig Routing() const {
    ops::RoutingConfig routing;
    routing.scoring = ops::RouterScoring::kSoftmaxTopkRenorm;
    routing.topk = num_experts_per_tok;
    routing.normalize = norm_topk_prob;
    return routing;
  }

  /// \brief Per-expert intermediate size under tensor parallelism.
  StatusOr<int64_t> ExpertIntermediateShard(int64_t tp_size) const {
    const int64_t full = base.intermediate_size;  // Per-expert width.
    if (tp_size <= 0 || full % tp_size != 0) {
      return InvalidArgumentError("expert intermediate (", full,
                                  ") must divide tensor-parallel size ", tp_size);
    }
    return full / tp_size;
  }
};

}  // namespace qwen3_moe
}  // namespace models
}  // namespace inferx

#endif  // INFERX_MODELS_QWEN3_MOE_QWEN3_MOE_CONFIG_H_
