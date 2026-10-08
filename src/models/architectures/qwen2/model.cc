#include "inferx/models/architectures/qwen2/model.h"

namespace inferx::models::qwen2 {

StatusOr<std::unique_ptr<Qwen2ForCausalLM>> Qwen2ForCausalLM::Build(
    const Qwen2Config& config, const ModelLoadOptions& options) {
  INFERX_ASSIGN_OR_RETURN(auto stack, BuildStackConfig(config));
  INFERX_ASSIGN_OR_RETURN(auto prepared,
                          PrepareAttentionDecoder(std::move(stack), CheckpointNames(),
                                                  CheckpointLayout(), "Qwen2ForCausalLM", config,
                                                  options));
  return std::unique_ptr<Qwen2ForCausalLM>(
      new Qwen2ForCausalLM(std::move(prepared), options.max_tokens, options.max_seqs));
}

}  // namespace inferx::models::qwen2
