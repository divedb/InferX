#include "inferx/models/architectures/deepseek_v3/model.h"

namespace inferx::models::deepseek_v3 {

StatusOr<std::unique_ptr<DeepseekV3ForCausalLM>> DeepseekV3ForCausalLM::Build(
    const DeepseekV3Config& config, const ModelLoadOptions& options) {
  INFERX_ASSIGN_OR_RETURN(auto stack, BuildStackConfig(config));
  INFERX_ASSIGN_OR_RETURN(auto prepared,
                          PrepareAttentionDecoder(std::move(stack), CheckpointNames(),
                                                  CheckpointLayout(), "DeepseekV3ForCausalLM",
                                                  config, options));
  return std::unique_ptr<DeepseekV3ForCausalLM>(
      new DeepseekV3ForCausalLM(std::move(prepared), options.max_tokens, options.max_seqs));
}

}  // namespace inferx::models::deepseek_v3
