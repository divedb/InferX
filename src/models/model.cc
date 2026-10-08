#include "inferx/models/model.h"

#include "inferx/core/logging.h"
#include "inferx/models/registry.h"
#include "inferx/util/hf_config.h"

namespace inferx::models {

StatusOr<StepExecutionInfo> CausalLanguageModel::PrepareStep(
    const ModelInputs& inputs, const LogitSelection& selection, ForwardContext& context) {
  const auto& batch = inputs.attention;
  if (batch.num_tokens <= 0 || batch.num_seqs <= 0 ||
      inputs.token_ids.Numel() != batch.num_tokens) {
    return InvalidArgumentError("step geometry is inconsistent");
  }
  if (selection.rows.Numel() > batch.num_tokens) {
    return InvalidArgumentError("logit selection exceeds the token batch");
  }
  if (context.state.layers.empty()) {
    return InvalidArgumentError("no persistent state is bound to the model");
  }
  return StepExecutionInfo{};
}

StatusOr<std::unique_ptr<CausalLanguageModel>> LoadModel(const ModelLoadOptions& options) {
  INFERX_LOG(INFO) << "loading model from " << options.directory;
  // Identity first: an unsupported architecture is rejected before any
  // weight shard is mapped.
  INFERX_ASSIGN_OR_RETURN(auto hf, HfConfig::FromFile(options.directory + "/config.json"));
  INFERX_ASSIGN_OR_RETURN(const ArchitectureEntry* entry,
                          ArchitectureRegistry::Builtins().Resolve(hf.ModelType(),
                                                                   hf.Architecture()));
  INFERX_LOG(INFO) << "architecture " << entry->architecture << " (" << hf.ModelType() << ")";
  // The factory parses the text-model view (text_config for multimodal
  // wrappers, otherwise the root) and owns the rest of the lifecycle:
  // validation, weights, and execution resources.
  return entry->build(hf.Text(), options);
}

}  // namespace inferx::models
