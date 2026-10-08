/// \file
/// \brief Llama causal language model: the canonical pre-norm decoder
/// composition (RMSNorm + GQA attention without Q/K norm, RMSNorm +
/// gated MLP per layer). Validates reuse of the shared layers beyond
/// Qwen3.

#ifndef INFERX_MODELS_ARCHITECTURES_LLAMA_MODEL_H_
#define INFERX_MODELS_ARCHITECTURES_LLAMA_MODEL_H_

#include <memory>

#include "inferx/core/status.h"
#include "inferx/models/architectures/llama/config.h"
#include "inferx/models/decoder_lm.h"
#include "inferx/models/load_options.h"
#include "inferx/models/loading/weight_mapping.h"

namespace inferx::models::llama {

class LlamaForCausalLM final : public DecoderLm {
 public:
  using Config = LlamaConfig;

  static StatusOr<std::unique_ptr<LlamaForCausalLM>> Build(const LlamaConfig& config,
                                                           const ModelLoadOptions& options);

 private:
  LlamaForCausalLM(DecoderLm::Prepared prepared, int max_tokens, int max_seqs)
      : DecoderLm(std::move(prepared), max_tokens, max_seqs) {}
};

/// Assembles the decoder-stack configuration (weights.cc).
StatusOr<layers::StackConfig> BuildStackConfig(const LlamaConfig& config);

const WeightNames& CheckpointNames();
const WeightLayout& CheckpointLayout();

}  // namespace inferx::models::llama

#endif  // INFERX_MODELS_ARCHITECTURES_LLAMA_MODEL_H_
