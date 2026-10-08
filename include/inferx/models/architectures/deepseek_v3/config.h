/// \file
/// \brief DeepSeek-V3 configuration: multi-latent attention in every
/// layer, sigmoid/grouped expert routing past a dense prefix. Owns its
/// MLA and MoE assembly.

#ifndef INFERX_MODELS_ARCHITECTURES_DEEPSEEK_V3_CONFIG_H_
#define INFERX_MODELS_ARCHITECTURES_DEEPSEEK_V3_CONFIG_H_

#include <cstdint>

#include "inferx/core/status.h"
#include "inferx/models/configs/dims.h"
#include "inferx/models/configs/rope.h"
#include "inferx/util/config_view.h"

namespace inferx::models::deepseek_v3 {

struct DeepseekV3Config : ModelDims {
  /// MLA geometry; the rope window is the per-head rope width.
  RopeConfig rope;
  int64_t q_lora_rank = 0;
  int64_t kv_lora_rank = 0;
  int64_t qk_nope_head_dim = 0;
  int64_t qk_rope_head_dim = 0;
  int64_t v_head_dim = 0;

  /// Routed experts (sigmoid/grouped top-k with a selection-only
  /// correction bias and a dense shared expert of the model width).
  int64_t n_routed_experts = 0;
  int64_t num_experts_per_tok = 0;
  int64_t moe_intermediate_size = 0;
  bool norm_topk_prob = true;
  double routed_scaling_factor = 1.0;
  int64_t n_group = 1;
  int64_t topk_group = 0;  ///< 0 derives n_group.
  /// Layers [0, first_k_dense_replace) keep a dense feed-forward.
  int64_t first_k_dense_replace = 0;

  static StatusOr<DeepseekV3Config> Parse(const ConfigView& view);
  Status Validate() const;

  /// Combined per-head key dimension; also the padded V dimension.
  int64_t head_dim() const { return qk_nope_head_dim + qk_rope_head_dim; }
};

}  // namespace inferx::models::deepseek_v3

#endif  // INFERX_MODELS_ARCHITECTURES_DEEPSEEK_V3_CONFIG_H_
