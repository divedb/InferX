#ifndef INFERX_CONFIG_MODEL_CONFIG_H_
#define INFERX_CONFIG_MODEL_CONFIG_H_

#include <cstdint>
#include <string>

#include "inferx/config/device_config.h"

namespace inferx {

struct ModelConfig {
  /// Directory holding the model checkpoint.
  /// EXAMPLE: --model /path/to/checkpoint
  std::string model_dir;

  /// Directory holding tokenizer.json; empty derives from model_dir via ResolvedTokenizerDir().
  /// EXAMPLE: --tokenizer /path/to/tokenizer
  std::string tokenizer_dir;

  /// Data type for weights and activations; "auto" (the default) uses the checkpoint's dtype.
  /// EXAMPLE: --dtype bfloat16
  std::string dtype = "auto";

  /// Random seed for reproducible sampling.
  /// EXAMPLE: --seed 42
  std::int64_t seed = 0;

  /// Maximum sequence length (prompt plus output); 0 keeps the checkpoint's
  /// context limit.
  /// EXAMPLE: --max-model-len 8192
  std::int64_t max_model_len = 0;

  /// Name the API reports for this model; empty falls back to model_dir
  /// through ServedName().
  /// EXAMPLE: --served-model-name inferx-8b
  std::string served_model_name;

  /// Generation config to use; "auto" (the default) uses the checkpoint's generation config,
  /// e.g. generation_config.json.
  /// EXAMPLE: --generation-config /path/to/generation_config.json
  std::string generation_config = "auto";

  /// Generation config to override the checkpoint's generation config; empty uses the
  /// checkpoint's generation config.
  /// EXAMPLE: --override-generation-config '{"temperature": 0.5}'
  std::string override_generation_config;

  /// \brief The name the API reports for this model; falls back to model_dir if
  ///        served_model_name is empty.
  ///
  /// \return The name the API reports for this model.
  std::string ServedName() const {
    return served_model_name.empty() ? model_dir : served_model_name;
  }

  /// \brief The directory holding tokenizer.json; falls back to model_dir if tokenizer_dir is
  ///        empty.
  ///
  /// \return The directory holding tokenizer.json.
  std::string ResolvedTokenizerDir() const {
    return tokenizer_dir.empty() ? model_dir : tokenizer_dir;
  }
};

}  // namespace inferx

#endif  // INFERX_CONFIG_MODEL_CONFIG_H_
