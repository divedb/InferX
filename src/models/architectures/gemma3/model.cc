#include "inferx/models/architectures/gemma3/model.h"

namespace inferx::models::gemma3 {

StatusOr<std::unique_ptr<Gemma3ForCausalLM>> Gemma3ForCausalLM::Build(
    const Gemma3Config& config, const ModelLoadOptions& options) {
  INFERX_ASSIGN_OR_RETURN(auto stack, BuildStackConfig(config));
  INFERX_ASSIGN_OR_RETURN(auto prepared,
                          PrepareAttentionDecoder(std::move(stack), CheckpointNames(),
                                                  CheckpointLayout(), "Gemma3ForCausalLM", config,
                                                  options));
  return std::unique_ptr<Gemma3ForCausalLM>(
      new Gemma3ForCausalLM(std::move(prepared), options.max_tokens, options.max_seqs));
}

}  // namespace inferx::models::gemma3
