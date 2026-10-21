/// \file
/// \brief Generation settings read from a checkpoint's
/// generation_config.json. Serving and generation configuration belongs
/// to the engine and serving layer; the models layer never sees it.

#ifndef INFERX_SERVER_GENERATION_CONFIG_H_
#define INFERX_SERVER_GENERATION_CONFIG_H_

#include <cstdint>
#include <optional>
#include <string>

#include "inferx/core/status.h"

namespace inferx {

/// \brief Sampling defaults a checkpoint ships for itself. Absent fields
///        stay unset so engine defaults win.
struct GenerationConfig {
  std::optional<float> temperature;
  std::optional<float> top_p;
  std::optional<float> min_p;
  std::optional<std::int64_t> top_k;  ///< HF -1 means disabled.
  std::optional<float> repetition_penalty;
  std::optional<std::int64_t> max_new_tokens;

  static StatusOr<GenerationConfig> FromJson(const std::string& text);
  static StatusOr<GenerationConfig> FromFile(const std::string& path);

  /// \brief Resolves the generation config a run uses.
  ///
  /// \param model_dir          Checkpoint directory (mode "auto").
  /// \param generation_config  "auto", "vllm" (none), or an explicit path.
  /// \return                   The config, nullopt when none applies, or an
  ///                           error (a missing explicit file is an error;
  ///                           a missing auto file is not).
  static StatusOr<std::optional<GenerationConfig>> Resolve(const std::string& model_dir,
                                                           const std::string& generation_config);
};

}  // namespace inferx

#endif  // INFERX_SERVER_GENERATION_CONFIG_H_
