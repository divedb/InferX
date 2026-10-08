#include "inferx/models/architectures/mistral/model.h"

namespace inferx::models::mistral {

StatusOr<std::unique_ptr<MistralForCausalLM>> MistralForCausalLM::Build(
    const MistralConfig& config, const ModelLoadOptions& options) {
  INFERX_ASSIGN_OR_RETURN(auto stack, BuildStackConfig(config));
  INFERX_ASSIGN_OR_RETURN(auto prepared,
                          PrepareAttentionDecoder(std::move(stack), CheckpointNames(),
                                                  CheckpointLayout(), "MistralForCausalLM", config,
                                                  options));
  return std::unique_ptr<MistralForCausalLM>(
      new MistralForCausalLM(std::move(prepared), options.max_tokens, options.max_seqs));
}

}  // namespace inferx::models::mistral
