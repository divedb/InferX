#include "inferx/models/qwen3_moe/qwen3_moe_config.h"

namespace inferx {
namespace models {
namespace qwen3_moe {

StatusOr<Qwen3MoeConfig> Qwen3MoeConfig::FromHf(const HfConfig& config) {
  INFERX_ASSIGN_OR_RETURN(qwen3::Qwen3Config base, qwen3::Qwen3Config::FromHf(config));

  const ConfigView& text = config.Text();
  Qwen3MoeConfig parsed;
  parsed.base = std::move(base);

  INFERX_ASSIGN_OR_RETURN(parsed.num_experts, text.RequireInt("num_experts"));
  INFERX_ASSIGN_OR_RETURN(parsed.num_experts_per_tok,
                          text.RequireInt("num_experts_per_tok"));
  INFERX_ASSIGN_OR_RETURN(parsed.norm_topk_prob, text.Bool("norm_topk_prob", true));
  INFERX_ASSIGN_OR_RETURN(parsed.shared_expert_intermediate,
                          text.Int("shared_expert_intermediate_size", 0));
  INFERX_ASSIGN_OR_RETURN(parsed.shared_expert_gate, text.Bool("shared_expert_gate", false));

  if (parsed.num_experts <= 0 || parsed.num_experts_per_tok <= 0 ||
      parsed.num_experts_per_tok > parsed.num_experts) {
    return InvalidArgumentError("qwen3-moe expert counts are inconsistent: ",
                                parsed.num_experts, " experts, ",
                                parsed.num_experts_per_tok, " per token");
  }
  if (parsed.shared_expert_intermediate < 0) {
    return InvalidArgumentError("qwen3-moe shared expert intermediate must be >= 0");
  }
  return parsed;
}

}  // namespace qwen3_moe
}  // namespace models
}  // namespace inferx
