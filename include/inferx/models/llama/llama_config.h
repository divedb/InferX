#ifndef INFERX_MODELS_LLAMA_LLAMA_CONFIG_H_
#define INFERX_MODELS_LLAMA_LLAMA_CONFIG_H_

#include <cstdint>

#include "inferx/core/status.h"
#include "inferx/models/requirements.h"
#include "inferx/ops/rotary.h"
#include "inferx/tokenizer/id.h"
#include "inferx/util/hf_config.h"

namespace inferx {
namespace models {
namespace llama {

/// \brief The Llama architecture's checked `config.json` fields.
struct LlamaConfig {
  int64_t vocab_size = 0;
  int64_t hidden_size = 0;
  int64_t num_hidden_layers = 0;
  int64_t num_attention_heads = 0;
  int64_t num_key_value_heads = 0;
  int64_t head_dim = 0;
  int64_t intermediate_size = 0;
  int64_t max_position_embeddings = 0;
  int64_t sliding_window = 0;  ///< 0: full attention.
  float rms_norm_eps = 1e-5f;
  bool tie_word_embeddings = false;
  std::vector<TokenId> eos_token_ids;
  ops::RotaryParams rope;

  static StatusOr<LlamaConfig> FromHf(const HfConfig& config);

  StatusOr<int64_t> LocalHeads(int64_t tp_size) const {
    if (tp_size <= 0 || num_attention_heads % tp_size != 0) {
      return InvalidArgumentError("attention heads (", num_attention_heads,
                                  ") must divide tensor-parallel size ", tp_size);
    }
    return num_attention_heads / tp_size;
  }

  StatusOr<int64_t> LocalKvHeads(int64_t tp_size) const {
    if (tp_size <= 0 || num_key_value_heads % tp_size != 0) {
      return InvalidArgumentError("kv heads (", num_key_value_heads,
                                  ") must divide tensor-parallel size ", tp_size);
    }
    return num_key_value_heads / tp_size;
  }

  KvLayout KvLayoutFor(int64_t tp_size) const {
    inferx::KvLayout layout;
    layout.kv_heads = num_key_value_heads / tp_size;
    layout.head_dim = head_dim;
    return layout;
  }
};

}  // namespace llama
}  // namespace models
}  // namespace inferx

#endif  // INFERX_MODELS_LLAMA_LLAMA_CONFIG_H_
