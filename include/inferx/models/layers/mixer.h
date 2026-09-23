/// \file
/// \brief Token mixers: the component that mixes information across the
/// sequence between a block's two norms.

#ifndef INFERX_MODELS_LAYERS_MIXER_H_
#define INFERX_MODELS_LAYERS_MIXER_H_

#include <cstdint>
#include <string>

#include "inferx/core/tensor.h"

namespace inferx::layers {

/// \brief Extra learned output projection attached to attention.
enum class OutputGate {
  kNone,     ///< Plain attention output.
  kSigmoid,  ///< Sigmoid-gated output; doubles q_proj rows (Qwen3-Next).
};

/// \brief Rotary position embedding applied inside an attention mixer.
struct RotaryConfig {
  int64_t dim = 0;               ///< Rotated columns per head; even, <= head_dim.
  float theta = 10000.0f;        ///< Base frequency.
  float factor = 1.0f;           ///< Scaling factor applied by the flavor.
  std::string type = "default";  ///< Flavor: default, yarn, linear, ...
  /// Raw scaling JSON for flavors beyond the shared fields.
  std::string parameters_json = "{}";
};

/// \brief Grouped-query causal attention.
struct AttentionConfig {
  int64_t query_heads = 0;         ///< Total query heads, before any sharding.
  int64_t kv_heads = 0;            ///< Key/value heads; divides query_heads.
  int64_t head_dim = 0;
  bool qk_norm = false;            ///< Per-head RMSNorm on q and k.
  bool projection_bias = false;    ///< Biases on the q/k/v/o projections.
  RotaryConfig rotary;
  int64_t sliding_window = 0;      ///< Tokens; 0 disables windowing.
  OutputGate output_gate = OutputGate::kNone;
};

/// \brief Gated DeltaNet linear attention (Qwen3-Next recurrent layers).
struct GatedDeltaNetConfig {
  int64_t key_heads = 0;
  int64_t value_heads = 0;
  int64_t key_dim = 0;
  int64_t value_dim = 0;
  int64_t conv_kernel_size = 0;
};

/// \brief One affine projection, stored as the checkpoint stores it.
struct LinearWeights {
  Tensor weight;  ///< [out, in]
  Tensor bias;    ///< [out]; undefined when the projection is bias-free.
};

/// \brief Attention projections; shapes follow AttentionConfig.
struct AttentionWeights {
  Tensor packed_qkv;     ///< Optional concatenated projection rows.
  LinearWeights query;   ///< [query_heads * head_dim (* 2 when gated), hidden]
  LinearWeights key;     ///< [kv_heads * head_dim, hidden]
  LinearWeights value;   ///< [kv_heads * head_dim, hidden]
  LinearWeights output;  ///< [hidden, query_heads * head_dim]
  Tensor query_norm;     ///< [head_dim]; undefined unless qk_norm.
  Tensor key_norm;       ///< [head_dim]; undefined unless qk_norm.
};

}  // namespace inferx::layers

#endif  // INFERX_MODELS_LAYERS_MIXER_H_
