/// \file
/// \brief Mistral causal language model: pre-norm decoder with optional
/// sliding-window attention.

#ifndef INFERX_MODELS_ARCHITECTURES_MISTRAL_MODEL_H_
#define INFERX_MODELS_ARCHITECTURES_MISTRAL_MODEL_H_

#include <memory>

#include "inferx/core/status.h"
#include "inferx/models/architectures/mistral/config.h"
#include "inferx/models/decoder_lm.h"
#include "inferx/models/load_options.h"
#include "inferx/models/loading/weight_mapping.h"

namespace inferx::models::mistral {

class MistralForCausalLM final : public DecoderLm {
 public:
  using Config = MistralConfig;

  static StatusOr<std::unique_ptr<MistralForCausalLM>> Build(const MistralConfig& config,
                                                             const ModelLoadOptions& options);

 private:
  MistralForCausalLM(DecoderLm::Prepared prepared, int max_tokens, int max_seqs)
      : DecoderLm(std::move(prepared), max_tokens, max_seqs) {}
};

StatusOr<layers::StackConfig> BuildStackConfig(const MistralConfig& config);

const WeightNames& CheckpointNames();
const WeightLayout& CheckpointLayout();

}  // namespace inferx::models::mistral

#endif  // INFERX_MODELS_ARCHITECTURES_MISTRAL_MODEL_H_
