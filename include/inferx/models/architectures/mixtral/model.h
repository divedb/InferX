/// \file
/// \brief Mixtral causal language model: pre-norm decoder whose every
/// feed-forward is a softmax top-k mixture of experts.

#ifndef INFERX_MODELS_ARCHITECTURES_MIXTRAL_MODEL_H_
#define INFERX_MODELS_ARCHITECTURES_MIXTRAL_MODEL_H_

#include <memory>

#include "inferx/core/status.h"
#include "inferx/models/architectures/mixtral/config.h"
#include "inferx/models/decoder_lm.h"
#include "inferx/models/load_options.h"
#include "inferx/models/loading/weight_mapping.h"

namespace inferx::models::mixtral {

class MixtralForCausalLM final : public DecoderLm {
 public:
  using Config = MixtralConfig;

  static StatusOr<std::unique_ptr<MixtralForCausalLM>> Build(const MixtralConfig& config,
                                                             const ModelLoadOptions& options);

 private:
  MixtralForCausalLM(DecoderLm::Prepared prepared, int max_tokens, int max_seqs)
      : DecoderLm(std::move(prepared), max_tokens, max_seqs) {}
};

StatusOr<layers::StackConfig> BuildStackConfig(const MixtralConfig& config);

const WeightNames& CheckpointNames();
const WeightLayout& CheckpointLayout();

}  // namespace inferx::models::mixtral

#endif  // INFERX_MODELS_ARCHITECTURES_MIXTRAL_MODEL_H_
