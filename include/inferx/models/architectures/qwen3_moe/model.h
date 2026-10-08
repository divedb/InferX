/// \file
/// \brief Qwen3-MoE causal language model: Qwen3 attention with routed
/// experts on the sparse-step schedule.

#ifndef INFERX_MODELS_ARCHITECTURES_QWEN3_MOE_MODEL_H_
#define INFERX_MODELS_ARCHITECTURES_QWEN3_MOE_MODEL_H_

#include <memory>

#include "inferx/core/status.h"
#include "inferx/models/architectures/qwen3_moe/config.h"
#include "inferx/models/decoder_lm.h"
#include "inferx/models/load_options.h"
#include "inferx/models/loading/weight_mapping.h"

namespace inferx::models::qwen3_moe {

class Qwen3MoeForCausalLM final : public DecoderLm {
 public:
  using Config = Qwen3MoeConfig;

  static StatusOr<std::unique_ptr<Qwen3MoeForCausalLM>> Build(const Qwen3MoeConfig& config,
                                                              const ModelLoadOptions& options);

 private:
  Qwen3MoeForCausalLM(DecoderLm::Prepared prepared, int max_tokens, int max_seqs)
      : DecoderLm(std::move(prepared), max_tokens, max_seqs) {}
};

StatusOr<layers::StackConfig> BuildStackConfig(const Qwen3MoeConfig& config);

const WeightNames& CheckpointNames();
const WeightLayout& CheckpointLayout();

}  // namespace inferx::models::qwen3_moe

#endif  // INFERX_MODELS_ARCHITECTURES_QWEN3_MOE_MODEL_H_
