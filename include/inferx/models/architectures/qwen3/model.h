/// \file
/// \brief Qwen3 dense causal language model. Owns its composition:
///
/// \code
/// Qwen3ForCausalLM
/// ├── Qwen3Model
/// │   ├── TokenEmbedding
/// │   ├── Qwen3DecoderLayer × N
/// │   │   ├── RMSNorm
/// │   │   ├── Qwen3Attention (QKV, Q/K norm, RoPE, attention, output)
/// │   │   ├── RMSNorm
/// │   │   └── GatedMLP
/// │   └── RMSNorm
/// └── LMHead
/// \endcode
///
/// MoE (`Qwen3MoeConfig`) and recurrent-hybrid (`Qwen3NextConfig`)
/// variants are separate architectures.

#ifndef INFERX_MODELS_ARCHITECTURES_QWEN3_MODEL_H_
#define INFERX_MODELS_ARCHITECTURES_QWEN3_MODEL_H_

#include <memory>

#include "inferx/core/status.h"
#include "inferx/models/architectures/qwen3/config.h"
#include "inferx/models/decoder_lm.h"
#include "inferx/models/load_options.h"
#include "inferx/models/loading/weight_mapping.h"

namespace inferx::models::qwen3 {

class Qwen3ForCausalLM final : public DecoderLm {
 public:
  using Config = Qwen3Config;

  /// \brief Factory: validates execution support, loads weights, and
  ///        initializes execution resources. Registered under
  ///        "Qwen3ForCausalLM" / "qwen3".
  static StatusOr<std::unique_ptr<Qwen3ForCausalLM>> Build(const Qwen3Config& config,
                                                           const ModelLoadOptions& options);

 private:
  Qwen3ForCausalLM(DecoderLm::Prepared prepared, int max_tokens, int max_seqs)
      : DecoderLm(std::move(prepared), max_tokens, max_seqs) {}
};

/// \brief Assembles the shared decoder-stack configuration: per-layer
///        Qwen3 attention (Q/K RMSNorm + RoPE) and SiLU-gated MLP, plus the
///        closing norm. Implemented in weights.cc alongside the checkpoint
///        mapping.
StatusOr<layers::StackConfig> BuildStackConfig(const Qwen3Config& config);

/// Qwen3's checkpoint tensor names and fusion layout (weights.cc).
const WeightNames& CheckpointNames();
const WeightLayout& CheckpointLayout();

}  // namespace inferx::models::qwen3

#endif  // INFERX_MODELS_ARCHITECTURES_QWEN3_MODEL_H_
