/// \file
/// \brief ModelDims: shared decoder dimensions every causal architecture
/// embeds in its typed configuration. One parse implementation keeps the
/// common keys consistent; each architecture decides which of the derived
/// fields and defaults are valid for it and adds its own fields.

#ifndef INFERX_MODELS_CONFIGS_DIMS_H_
#define INFERX_MODELS_CONFIGS_DIMS_H_

#include <cstdint>
#include <string>

#include "inferx/core/status.h"
#include "inferx/ops/elementwise.h"
#include "inferx/util/config_view.h"

namespace inferx::models {

/// \brief Resolves the feed-forward activation a config selects:
///        `hidden_act` (silu, gelu_pytorch_tanh/geglu), with swiglu_limit >
///        0 selecting the gpt-oss SwiGLU-oai flavor regardless of the
///        stated name. Unrecognized activations fail here.
StatusOr<ops::Activation> ParseActivation(const ConfigView& view);

/// \brief Global model dimensions as the checkpoint states them. These stay
///        immutable global values; rank-local dimensions are derived
///        separately during construction.
struct ModelDims {
  /// The model's hidden dimension, the size of every token's hidden state vector.
  int64_t hidden_size = 0;

  /// The model's intermediate dimension, the size of the feed-forward layer.
  int64_t intermediate_size = 0;

  /// The model's number of transformer layers, the depth of the decoder stack.
  int64_t num_hidden_layers = 0;

  /// The model's number of attention heads, the width of the multi-head attention.
  int64_t num_attention_heads = 0;

  /// Defaults to `num_attention_heads` when the checkpoint omits it.
  int64_t num_key_value_heads = 0;

  /// Defaults to `hidden_size / num_attention_heads` when omitted or zero.
  int64_t head_dim = 0;

  int64_t vocab_size = 0;
  /// Defaults to 2048, matching the common HF convention.
  int64_t max_position_embeddings = 2048;
  double rms_norm_eps = 1e-5;
  /// Resolved feed-forward activation (from `hidden_act`/`swiglu_limit`).
  ops::Activation activation = ops::Activation::kSilu;
  /// gpt-oss SwiGLU-oai clamp; 0 leaves the default.
  float swiglu_limit = 0.0f;
  /// True biases the Q/K/V projections (`attention_bias` in config.json).
  bool attention_bias = false;
  bool tie_word_embeddings = false;
  int64_t bos_token_id = -1;
  int64_t eos_token_id = -1;

  /// \brief Parses the shared dimensions from one architecture config view.
  ///
  /// Required keys missing produce errors naming the full field path;
  /// malformed values never fall back to defaults.
  static StatusOr<ModelDims> Parse(const ConfigView& view);

  /// \brief Checks that the dimensions are positive and self-consistent.
  Status Validate() const;

  int64_t QueryDim() const { return num_attention_heads * head_dim; }
  int64_t KvDim() const { return num_key_value_heads * head_dim; }
};

}  // namespace inferx::models

#endif  // INFERX_MODELS_CONFIGS_DIMS_H_
