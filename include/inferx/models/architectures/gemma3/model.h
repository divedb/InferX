/// \file
/// \brief Gemma3 text causal language model: sandwich-norm decoder
/// (each sublayer output normalized before joining the residual),
/// scaled embeddings, and alternating local/global attention.

#ifndef INFERX_MODELS_ARCHITECTURES_GEMMA3_MODEL_H_
#define INFERX_MODELS_ARCHITECTURES_GEMMA3_MODEL_H_

#include <memory>

#include "inferx/core/status.h"
#include "inferx/models/architectures/gemma3/config.h"
#include "inferx/models/decoder_lm.h"
#include "inferx/models/load_options.h"
#include "inferx/models/loading/weight_mapping.h"

namespace inferx::models::gemma3 {

class Gemma3ForCausalLM final : public DecoderLm {
 public:
  using Config = Gemma3Config;

  static StatusOr<std::unique_ptr<Gemma3ForCausalLM>> Build(const Gemma3Config& config,
                                                            const ModelLoadOptions& options);

 private:
  Gemma3ForCausalLM(DecoderLm::Prepared prepared, int max_tokens, int max_seqs)
      : DecoderLm(std::move(prepared), max_tokens, max_seqs) {}
};

StatusOr<layers::StackConfig> BuildStackConfig(const Gemma3Config& config);

const WeightNames& CheckpointNames();
const WeightLayout& CheckpointLayout();

}  // namespace inferx::models::gemma3

#endif  // INFERX_MODELS_ARCHITECTURES_GEMMA3_MODEL_H_
