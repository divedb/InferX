/// \file
/// \brief DecoderLm: the shared composition shell for architectures built
/// on the pre-norm decoder stack -- token embedding, stack, language-model
/// head -- behind the CausalLanguageModel boundary. Architecture models
/// construct it from their typed configurations; models whose execution
/// order differs may implement the boundary directly.
///
/// The shell does not define the model contract; CausalLanguageModel does.

#ifndef INFERX_MODELS_DECODER_LM_H_
#define INFERX_MODELS_DECODER_LM_H_

#include <memory>
#include <string>
#include <string_view>

#include "inferx/core/status.h"
#include "inferx/core/tensor.h"
#include "inferx/models/configs/dims.h"
#include "inferx/models/layers/decoder.h"
#include "inferx/models/layers/lm_head.h"
#include "inferx/models/loading/weight_mapping.h"
#include "inferx/models/load_options.h"
#include "inferx/models/model.h"
#include "inferx/models/model_info.h"
#include "inferx/models/requirements.h"

namespace inferx::models {

/// \brief Shared execution shell: embedding + decoder stack + LM head.
///        Architectures subclass it and own construction in their Build.
class DecoderLm : public CausalLanguageModel {
 public:
  /// Everything Build produces for the shell to execute.
  struct Prepared {
    ModelInfo info;              ///< Engine-facing facts.
    layers::StackConfig config;  ///< Rank-local execution configuration.
    layers::StackWeights weights;
    /// [vocab_shard, hidden]; may alias the token embedding when tied.
    Tensor head;
  };

  DecoderLm(Prepared prepared, int max_tokens, int max_seqs);

  const ModelInfo& info() const override { return info_; }
  const ModelRequirements& requirements() const override { return requirements_; }
  Status Prepare(const ExecutionLimits& limits, ForwardContext& context) override;
  MemoryUsage memory_usage() const override;
  StatusOr<StepExecutionInfo> PrepareStep(const ModelInputs& inputs,
                                          const LogitSelection& selection,
                                          ForwardContext& context) override;
  StatusOr<HiddenStates> Forward(const ModelInputs& inputs,
                                 ForwardContext& context) override;
  StatusOr<Logits> ComputeLogits(const HiddenStates& states, const LogitSelection& selection,
                                 ForwardContext& context) override;

 private:
  ModelInfo info_;
  ModelRequirements requirements_;
  layers::DecoderStack stack_;
  layers::LanguageModelHead head_;
  int max_seqs_;
};

/// \brief Shared lifecycle for attention-decoder architectures: validate
///        load policy and parallel support, load the mapped weights plus
///        head, and derive engine-facing facts. Called from each
///        architecture's Build after it assembles its (total) stack
///        configuration.
///
/// Factory steps covered: (2) execution-support validation for the
/// requested dtype and parallelism, (3/4) weight loading with shape and
/// completeness checks. `stack` carries TOTAL head counts; the returned
/// Prepared carries the rank-local copy.
StatusOr<DecoderLm::Prepared> PrepareAttentionDecoder(layers::StackConfig stack,
                                                      const WeightNames& names,
                                                      const WeightLayout& layout,
                                                      std::string_view architecture,
                                                      const ModelDims& dims,
                                                      const ModelLoadOptions& options);

}  // namespace inferx::models

#endif  // INFERX_MODELS_DECODER_LM_H_
