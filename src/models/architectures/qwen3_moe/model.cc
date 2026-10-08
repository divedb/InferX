#include "inferx/models/architectures/qwen3_moe/model.h"

namespace inferx::models::qwen3_moe {

StatusOr<std::unique_ptr<Qwen3MoeForCausalLM>> Qwen3MoeForCausalLM::Build(
    const Qwen3MoeConfig& config, const ModelLoadOptions& options) {
  INFERX_ASSIGN_OR_RETURN(auto stack, BuildStackConfig(config));
  INFERX_ASSIGN_OR_RETURN(auto prepared,
                          PrepareAttentionDecoder(std::move(stack), CheckpointNames(),
                                                  CheckpointLayout(), "Qwen3MoeForCausalLM", config,
                                                  options));
  return std::unique_ptr<Qwen3MoeForCausalLM>(
      new Qwen3MoeForCausalLM(std::move(prepared), options.max_tokens, options.max_seqs));
}

}  // namespace inferx::models::qwen3_moe
