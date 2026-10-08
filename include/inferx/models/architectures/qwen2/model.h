/// \file
/// \brief Qwen2 causal language model: pre-norm decoder with biased QKV
/// projections and SiLU-gated MLP.

#ifndef INFERX_MODELS_ARCHITECTURES_QWEN2_MODEL_H_
#define INFERX_MODELS_ARCHITECTURES_QWEN2_MODEL_H_

#include <memory>

#include "inferx/core/status.h"
#include "inferx/models/architectures/qwen2/config.h"
#include "inferx/models/decoder_lm.h"
#include "inferx/models/load_options.h"
#include "inferx/models/loading/weight_mapping.h"

namespace inferx::models::qwen2 {

class Qwen2ForCausalLM final : public DecoderLm {
 public:
  using Config = Qwen2Config;

  static StatusOr<std::unique_ptr<Qwen2ForCausalLM>> Build(const Qwen2Config& config,
                                                           const ModelLoadOptions& options);

 private:
  Qwen2ForCausalLM(DecoderLm::Prepared prepared, int max_tokens, int max_seqs)
      : DecoderLm(std::move(prepared), max_tokens, max_seqs) {}
};

StatusOr<layers::StackConfig> BuildStackConfig(const Qwen2Config& config);

const WeightNames& CheckpointNames();
const WeightLayout& CheckpointLayout();

}  // namespace inferx::models::qwen2

#endif  // INFERX_MODELS_ARCHITECTURES_QWEN2_MODEL_H_
