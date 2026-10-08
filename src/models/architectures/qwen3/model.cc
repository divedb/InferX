#include "inferx/models/architectures/qwen3/model.h"

namespace inferx::models::qwen3 {

StatusOr<std::unique_ptr<Qwen3ForCausalLM>> Qwen3ForCausalLM::Build(
    const Qwen3Config& config, const ModelLoadOptions& options) {
  INFERX_ASSIGN_OR_RETURN(auto stack, BuildStackConfig(config));
  INFERX_ASSIGN_OR_RETURN(auto prepared,
                          PrepareAttentionDecoder(std::move(stack), CheckpointNames(),
                                                  CheckpointLayout(), "Qwen3ForCausalLM", config,
                                                  options));
  return std::unique_ptr<Qwen3ForCausalLM>(
      new Qwen3ForCausalLM(std::move(prepared), options.max_tokens, options.max_seqs));
}

}  // namespace inferx::models::qwen3
