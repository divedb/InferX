#ifndef INFERX_MODELS_QWEN3_QWEN3_CONFIG_H_
#define INFERX_MODELS_QWEN3_QWEN3_CONFIG_H_

#include <cstdint>

#include "inferx/core/status.h"
#include "inferx/models/requirements.h"
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
  int64_t vocab_size = 0;
  int64_t hidden_size = 0;
  int64_t num_hidden_layers = 0;
  int64_t num_attention_heads = 0;
  int64_t num_key_value_heads = 0;
  int64_t head_dim = 0;
  int64_t intermediate_size = 0;
  int64_t max_position_embeddings = 0;
  int64_t sliding_window = 0;  ///< 0: full attention.
  float rms_norm_eps = 1e-6f;
  bool tie_word_embeddings = false;
  std::vector<TokenId> eos_token_ids;
  ops::RotaryParams rope;

  /// \brief Parses and validates the checkpoint's text config.
  static StatusOr<Qwen3Config> FromHf(const HfConfig& config);

  /// \brief Per-rank query heads under tensor parallelism.
  StatusOr<int64_t> LocalHeads(int64_t tp_size) const {
    if (tp_size <= 0 || num_attention_heads % tp_size != 0) {
      return InvalidArgumentError("attention heads (", num_attention_heads,
                                  ") must divide tensor-parallel size ", tp_size);
    }
    return num_attention_heads / tp_size;
  }

  /// \brief Per-rank KV heads under tensor parallelism.
  StatusOr<int64_t> LocalKvHeads(int64_t tp_size) const {
    if (tp_size <= 0 || num_key_value_heads % tp_size != 0) {
      return InvalidArgumentError("kv heads (", num_key_value_heads,
                                  ") must divide tensor-parallel size ", tp_size);
    }
    return num_key_value_heads / tp_size;
  }

  /// \brief This rank's paged-KV layout (kv heads sharded by TP).
  KvLayout KvLayoutFor(int64_t tp_size) const {
    inferx::KvLayout layout;
    layout.kv_heads = num_key_value_heads / tp_size;
    layout.head_dim = head_dim;
    return layout;
  }
};

}  // namespace qwen3
}  // namespace models
}  // namespace inferx

#endif  // INFERX_MODELS_QWEN3_QWEN3_CONFIG_H_
