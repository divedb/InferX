/// \file
/// \brief Qwen3-Next causal language model. Owns its heterogeneous
/// attention/recurrent layer schedule: full-attention layers gate their
/// output with a sigmoid projection, linear-attention layers run Gated
/// DeltaNet against the runner-allocated recurrent state, and every
/// sparse-step layer routes through experts with a gated shared expert.
/// Plus-one RMSNorm throughout.

#ifndef INFERX_MODELS_ARCHITECTURES_QWEN3_NEXT_MODEL_H_
#define INFERX_MODELS_ARCHITECTURES_QWEN3_NEXT_MODEL_H_

#include <memory>

#include "inferx/core/status.h"
#include "inferx/models/architectures/qwen3_next/config.h"
#include "inferx/models/decoder_lm.h"
#include "inferx/models/load_options.h"
#include "inferx/models/loading/weight_mapping.h"

namespace inferx::models::qwen3_next {

class Qwen3NextForCausalLM final : public DecoderLm {
 public:
  using Config = Qwen3NextConfig;

  static StatusOr<std::unique_ptr<Qwen3NextForCausalLM>> Build(
      const Qwen3NextConfig& config, const ModelLoadOptions& options);

 private:
  Qwen3NextForCausalLM(DecoderLm::Prepared prepared, int max_tokens, int max_seqs)
      : DecoderLm(std::move(prepared), max_tokens, max_seqs) {}
};

StatusOr<layers::StackConfig> BuildStackConfig(const Qwen3NextConfig& config);

const WeightNames& CheckpointNames();
const WeightLayout& CheckpointLayout();

}  // namespace inferx::models::qwen3_next

#endif  // INFERX_MODELS_ARCHITECTURES_QWEN3_NEXT_MODEL_H_
