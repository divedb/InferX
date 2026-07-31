#pragma once

#include <cstdint>
#include <string>
#include <string_view>

#include "inferx/core/dtype.h"
#include "inferx/core/status.h"

namespace inferx::model {

/// \brief The architecture family a checkpoint declares.
///
/// Qwen2 and Llama differ in exactly one thing that reaches the forward pass --
/// Qwen2 puts a bias on the Q/K/V projections and Llama does not -- so they are
/// one code path with a flag rather than two implementations. The enum exists
/// to reject architectures we have not read the definition of, rather than
/// running them as Llama and producing plausible nonsense.
enum class Architecture {
  kQwen2,
  kLlama,
};

const char* ArchitectureName(Architecture arch);

/// \brief A decoder-only transformer's shape, as read from `config.json`.
///
/// Every field is taken from the file; nothing is inferred except `head_dim`,
/// which HF omits when it is `hidden_size / num_attention_heads` and states
/// explicitly when it is not.
struct ModelConfig {
  Architecture architecture = Architecture::kQwen2;

  int64_t hidden_size = 0;
  int64_t intermediate_size = 0;
  int64_t num_hidden_layers = 0;
  int64_t num_attention_heads = 0;
  /// Fewer than `num_attention_heads` for GQA; equal for MHA.
  int64_t num_key_value_heads = 0;
  int64_t head_dim = 0;
  int64_t vocab_size = 0;
  int64_t max_position_embeddings = 0;

  double rms_norm_eps = 1e-6;
  double rope_theta = 10000.0;

  /// When true there is no `lm_head.weight`; the output projection reuses the
  /// embedding matrix. Qwen2.5-3B sets this.
  bool tie_word_embeddings = false;

  /// Qwen2 biases Q/K/V (never O); Llama biases nothing.
  bool attention_bias = false;

  /// The dtype the checkpoint stores, from `torch_dtype`. Usually BF16.
  DataType weight_dtype = DataType::kBFloat16;

  /// \brief Q heads per KV head. 1 means MHA.
  int64_t gqa_group_size() const {
    return num_key_value_heads > 0 ? num_attention_heads / num_key_value_heads
                                   : 0;
  }

  /// \brief Total width of the Q projection.
  int64_t q_dim() const { return num_attention_heads * head_dim; }

  /// \brief Total width of each of the K and V projections.
  int64_t kv_dim() const { return num_key_value_heads * head_dim; }

  /// \brief Checks the fields against each other.
  ///
  /// Separate from parsing because a config can be syntactically fine and still
  /// describe something impossible -- heads that do not divide, a GQA ratio
  /// that is not integral. Those are worth catching before the first weight is
  /// read rather than as a shape mismatch 36 layers in.
  Status Validate() const;

  std::string ToString() const;

  /// \brief Parses a `config.json` document.
  ///
  /// \param json_text The file's contents.
  /// \return          The config, already validated, or the first problem.
  static StatusOr<ModelConfig> FromJson(std::string_view json_text);

  /// \brief Reads and parses `<dir>/config.json`.
  static StatusOr<ModelConfig> FromDirectory(std::string_view dir);
};

}  // namespace inferx::model
