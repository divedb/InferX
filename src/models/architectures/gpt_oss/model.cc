#include "inferx/models/architectures/gpt_oss/model.h"

namespace inferx::models::gpt_oss {

StatusOr<std::unique_ptr<GptOssForCausalLM>> GptOssForCausalLM::Build(
    const GptOssConfig& config, const ModelLoadOptions& options) {
  INFERX_ASSIGN_OR_RETURN(auto stack, BuildStackConfig(config));
  INFERX_ASSIGN_OR_RETURN(auto prepared,
                          PrepareAttentionDecoder(std::move(stack), CheckpointNames(),
                                                  CheckpointLayout(), "GptOssForCausalLM", config,
                                                  options));
  return std::unique_ptr<GptOssForCausalLM>(
      new GptOssForCausalLM(std::move(prepared), options.max_tokens, options.max_seqs));
}

}  // namespace inferx::models::gpt_oss
