#include "inferx/models/architectures/llama/model.h"

namespace inferx::models::llama {

StatusOr<std::unique_ptr<LlamaForCausalLM>> LlamaForCausalLM::Build(
    const LlamaConfig& config, const ModelLoadOptions& options) {
  INFERX_ASSIGN_OR_RETURN(auto stack, BuildStackConfig(config));
  INFERX_ASSIGN_OR_RETURN(auto prepared,
                          PrepareAttentionDecoder(std::move(stack), CheckpointNames(),
                                                  CheckpointLayout(), "LlamaForCausalLM", config,
                                                  options));
  return std::unique_ptr<LlamaForCausalLM>(
      new LlamaForCausalLM(std::move(prepared), options.max_tokens, options.max_seqs));
}

}  // namespace inferx::models::llama
