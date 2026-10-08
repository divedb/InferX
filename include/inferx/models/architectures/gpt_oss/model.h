/// \file
/// \brief gpt-oss causal language model: pre-norm decoder whose every
/// layer routes through num_local_experts fused MXFP4 experts with
/// biased SwiGLU-oai activations, and whose attention carries per-head
/// sinks with windowed schedule support.

#ifndef INFERX_MODELS_ARCHITECTURES_GPT_OSS_MODEL_H_
#define INFERX_MODELS_ARCHITECTURES_GPT_OSS_MODEL_H_

#include <memory>

#include "inferx/core/status.h"
#include "inferx/models/architectures/gpt_oss/config.h"
#include "inferx/models/decoder_lm.h"
#include "inferx/models/load_options.h"
#include "inferx/models/loading/weight_mapping.h"

namespace inferx::models::gpt_oss {

class GptOssForCausalLM final : public DecoderLm {
 public:
  using Config = GptOssConfig;

  static StatusOr<std::unique_ptr<GptOssForCausalLM>> Build(const GptOssConfig& config,
                                                            const ModelLoadOptions& options);

 private:
  GptOssForCausalLM(DecoderLm::Prepared prepared, int max_tokens, int max_seqs)
      : DecoderLm(std::move(prepared), max_tokens, max_seqs) {}
};

StatusOr<layers::StackConfig> BuildStackConfig(const GptOssConfig& config);

const WeightNames& CheckpointNames();
const WeightLayout& CheckpointLayout();

}  // namespace inferx::models::gpt_oss

#endif  // INFERX_MODELS_ARCHITECTURES_GPT_OSS_MODEL_H_
