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

  DeviceConfig device;  ///< Placement (vLLM DeviceConfig).

  /// Data type for weights and activations; "auto" (the default) uses the checkpoint's dtype,
  /// or falls back to float32 if the checkpoint is ambiguous.
  ///
  std::string dtype = "auto";

  /// \brief Random seed for reproducible sampling.
  std::int64_t seed = 0;
  /// \brief Maximum sequence length (prompt plus output); 0 keeps the
  ///        checkpoint's context limit.
  std::int64_t max_model_len = 0;
  /// \brief Name the API reports for this model; empty falls back to
  ///        model_dir through ServedName() (vLLM: --served-model-name,
  ///        single name; vLLM also accepts a list).
  std::string served_model_name;
  /// \brief vLLM --generation-config: "auto" (the default) loads
  ///        generation_config.json from the model directory, "vllm" keeps
  ///        engine defaults, any other value is a directory to load it from.
  /// The mode string, not the parsed type -- that is models::GenerationConfig.
  std::string generation_config = "auto";
  /// \brief vLLM --override-generation-config: JSON merged over the
  ///        resolved generation config, e.g. `{"temperature": 0.5}`.
  std::string override_generation_config;

  /// \brief The name the API reports; --served-model-name or model_dir.
  std::string ServedName() const {
    return served_model_name.empty() ? model_dir : served_model_name;
  }

  /// \brief The directory holding tokenizer.json; --tokenizer or model_dir.
  std::string ResolvedTokenizerDir() const {
    return tokenizer_dir.empty() ? model_dir : tokenizer_dir;
  }
};

}  // namespace inferx

#endif  // INFERX_CONFIG_MODEL_CONFIG_H_
