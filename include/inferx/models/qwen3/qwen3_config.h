#ifndef INFERX_MODELS_QWEN3_QWEN3_CONFIG_H_
#define INFERX_MODELS_QWEN3_QWEN3_CONFIG_H_

#include <cstdint>

#include "inferx/core/status.h"
#include "inferx/ops/rotary.h"
#include "inferx/tokenizer/id.h"
#include "inferx/util/hf_config.h"

namespace inferx {
namespace models {
namespace qwen3 {

/// \brief The Qwen3 dense architecture's checked `config.json` fields.
///
/// Parsing happens once at load; everything afterwards reads these plain
/// values. Derived helpers carry the per-rank geometry so tensor-parallel
/// divisibility is checked in one place.
struct Qwen3Config {
  /// True when the q/k/v/o projections carry a bias term. False for all released dense Qwen3
  /// models.
  bool attention_bias = false;

  /// Dropout on attention probabilities. Always treated as 0 at inference.
  float attention_dropout = 0.0f;

  /// BOS token id; -1 if the checkpoint has none (JSON null or absent).
  TokenId bos_token_id = -1;

  /// EOS token ids. Parsed from `eos_token_id`, which may be a single int or a list.
  std::vector<TokenId> eos_token_ids;

  /// Per-head dimension. Set explicitly in the checkpoint and decoupled from hidden_size in
  /// Qwen3; must be even for RoPE.
  int64_t head_dim = 0;

  /// Activation used in the MLP gate. Qwen3 uses "silu"; FromHf rejects anything else the
  /// engine does not implement.
  std::string hidden_act = "silu";

  /// Width of the residual stream and token embeddings; also the output width of o_proj and
  /// down_proj. Not necessarily num_attention_heads * head_dim.
  int64_t hidden_size = 0;

  /// Stddev of the weight initializer used in training.
  float initializer_range = 0.02f;

  /// Inner width of the gated MLP (gate_proj / up_proj output, down_proj input).
  int64_t intermediate_size = 0;

  /// Maximum sequence length the checkpoint declares support for (without rope scaling).
  int64_t max_position_embeddings = 0;

  /// Layers at index >= this value use sliding-window attention when use_sliding_window is
  /// true. Ignored otherwise.
  int64_t max_window_layers = 0;

  /// Must be "qwen3"; used by FromHf to reject other architectures.
  std::string model_type = "qwen3";

  /// Number of query heads. The q_proj output width is num_attention_heads * head_dim, which
  /// may differ from hidden_size.
  int64_t num_attention_heads = 0;

  /// Number of transformer decoder layers.
  int64_t num_hidden_layers = 0;

  /// Number of key/value heads (grouped-query attention). Must divide num_attention_heads.
  int64_t num_key_value_heads = 0;

  /// Epsilon in every RMSNorm, including the per-head q_norm and k_norm that Qwen3 always
  /// applies to queries and keys before RoPE (no flag needed; they are part of the
  /// architecture).
  float rms_norm_eps = 1e-6f;

  /// RoPE parameters, including the base frequency and any scaling.
  ops::RotaryParams rotary_params;

  /// Window size in tokens; std::nullopt when the JSON value is null.
  std::optional<int64_t> sliding_window;

  /// True when lm_head shares weights with the embedding table (0.6B, 1.7B, 4B). When false the
  /// loader must read a separate `lm_head.weight` (8B, 14B, 32B). Always read from the file.
  bool tie_word_embeddings = true;

  /// Checkpoint weight dtype from `torch_dtype` / `dtype` (usually "bfloat16"); empty if
  /// absent.
  DataType dtype = DataType::kBFloat16;

  /// Whether HF generation keeps a KV cache by default. The engine manages its own cache.
  bool use_cache = true;

  /// True if sliding-window attention is enabled. False for released checkpoints.
  bool use_sliding_window = false;

  /// Number of rows in the embedding table (and in lm_head when untied).
  int64_t vocab_size = 0;

  /// \brief Parses and validates the checkpoint's text config.
  ///
  /// Validation includes: model_type == "qwen3", hidden_act is supported, head_dim > 0 and
  /// even, num_attention_heads % num_key_value_heads == 0, all dimensions positive, and EOS
  /// handled as int or list. Tolerates `rope_parameters` in place of `rope_scaling` and ignores
  /// unknown keys such as `layer_types`.
  ///
  /// \param config The HF config to parse.
  /// \return       The parsed config, or an error if the config is invalid.
  static StatusOr<Qwen3Config> FromHf(const HfConfig& config);
};

}  // namespace qwen3
}  // namespace models
}  // namespace inferx

#endif  // INFERX_MODELS_QWEN3_QWEN3_CONFIG_H_
