#include "inferx/models/architectures/mixtral/model.h"

namespace inferx::models::mixtral {

StatusOr<std::unique_ptr<MixtralForCausalLM>> MixtralForCausalLM::Build(
    const MixtralConfig& config, const ModelLoadOptions& options) {
  INFERX_ASSIGN_OR_RETURN(auto stack, BuildStackConfig(config));
  INFERX_ASSIGN_OR_RETURN(auto prepared,
                          PrepareAttentionDecoder(std::move(stack), CheckpointNames(),
                                                  CheckpointLayout(), "MixtralForCausalLM", config,
                                                  options));
  return std::unique_ptr<MixtralForCausalLM>(
      new MixtralForCausalLM(std::move(prepared), options.max_tokens, options.max_seqs));
}

}  // namespace inferx::models::mixtral
