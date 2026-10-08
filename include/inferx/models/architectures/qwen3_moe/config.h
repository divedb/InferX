/// \file
/// \brief Qwen3-MoE configuration: Qwen3 attention with every
/// sparse-step-th layer's feed-forward replaced by a routed mixture of
/// experts (optionally with a shared expert). Separate type from the
/// dense Qwen3Config.

#ifndef INFERX_MODELS_ARCHITECTURES_QWEN3_MOE_CONFIG_H_
#define INFERX_MODELS_ARCHITECTURES_QWEN3_MOE_CONFIG_H_

#include <cstdint>
#include <vector>

#include "inferx/core/status.h"
#include "inferx/models/configs/dims.h"
#include "inferx/models/configs/rope.h"
#include "inferx/util/config_view.h"

namespace inferx::models::qwen3_moe {

struct Qwen3MoeConfig : ModelDims {
  RopeConfig rope;
  int64_t num_experts = 0;
  int64_t num_experts_per_tok = 0;
  int64_t moe_intermediate_size = 0;
  /// Renormalize selected expert weights to sum to one.
  bool norm_topk_prob = true;
  /// 0 disables the shared expert.
  int64_t shared_expert_intermediate_size = 0;
  /// Every sparse-step-th layer routes; 1 routes in every layer.
  int64_t decoder_sparse_step = 1;
  /// Layer indices kept dense despite the sparse step.
  std::vector<int64_t> mlp_only_layers;

  static StatusOr<Qwen3MoeConfig> Parse(const ConfigView& view);
  Status Validate() const;

 protected:
  /// Shared parse core (dimensions, rope, expert fields); subclasses add
  /// their own layer-schedule keys on top.
  static StatusOr<Qwen3MoeConfig> ParseBase(const ConfigView& view);
};

}  // namespace inferx::models::qwen3_moe

#endif  // INFERX_MODELS_ARCHITECTURES_QWEN3_MOE_CONFIG_H_
