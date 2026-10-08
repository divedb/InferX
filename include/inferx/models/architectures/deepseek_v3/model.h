/// \file
/// \brief DeepSeek-V3 causal language model. Owns its MLA and MoE
/// assembly: multi-latent attention in every layer (decompressed into the
/// shared paged-attention path) and sigmoid/grouped expert routing with a
/// selection-only correction bias past the dense prefix.

#ifndef INFERX_MODELS_ARCHITECTURES_DEEPSEEK_V3_MODEL_H_
#define INFERX_MODELS_ARCHITECTURES_DEEPSEEK_V3_MODEL_H_

#include <memory>

#include "inferx/core/status.h"
#include "inferx/models/architectures/deepseek_v3/config.h"
#include "inferx/models/decoder_lm.h"
#include "inferx/models/load_options.h"
#include "inferx/models/loading/weight_mapping.h"

namespace inferx::models::deepseek_v3 {

class DeepseekV3ForCausalLM final : public DecoderLm {
 public:
  using Config = DeepseekV3Config;

  static StatusOr<std::unique_ptr<DeepseekV3ForCausalLM>> Build(
      const DeepseekV3Config& config, const ModelLoadOptions& options);

 private:
  DeepseekV3ForCausalLM(DecoderLm::Prepared prepared, int max_tokens, int max_seqs)
      : DecoderLm(std::move(prepared), max_tokens, max_seqs) {}
};

StatusOr<layers::StackConfig> BuildStackConfig(const DeepseekV3Config& config);

const WeightNames& CheckpointNames();
const WeightLayout& CheckpointLayout();

}  // namespace inferx::models::deepseek_v3

#endif  // INFERX_MODELS_ARCHITECTURES_DEEPSEEK_V3_MODEL_H_
