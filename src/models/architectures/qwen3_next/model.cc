#include "inferx/models/architectures/qwen3_next/model.h"

namespace inferx::models::qwen3_next {

StatusOr<std::unique_ptr<Qwen3NextForCausalLM>> Qwen3NextForCausalLM::Build(
    const Qwen3NextConfig& config, const ModelLoadOptions& options) {
  INFERX_ASSIGN_OR_RETURN(auto stack, BuildStackConfig(config));
  INFERX_ASSIGN_OR_RETURN(auto prepared,
                          PrepareAttentionDecoder(std::move(stack), CheckpointNames(),
                                                  CheckpointLayout(), "Qwen3NextForCausalLM",
                                                  config, options));
  return std::unique_ptr<Qwen3NextForCausalLM>(
      new Qwen3NextForCausalLM(std::move(prepared), options.max_tokens, options.max_seqs));
}

}  // namespace inferx::models::qwen3_next
