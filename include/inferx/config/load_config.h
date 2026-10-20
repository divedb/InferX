#ifndef INFERX_CONFIG_LOAD_CONFIG_H_
#define INFERX_CONFIG_LOAD_CONFIG_H_

#include <string>
#include <vector>

#include "inferx/core/status.h"

namespace inferx {

/// \brief Checkpoint weight format (vLLM: --load-format).
enum class LoadFormat {
  kAuto,  ///< Prefer safetensors, fall back to pt.
  kSafetensors,
  kPt,
  kSharded,  ///< Per-rank sharded checkpoint.
  kDummy,    ///< Random weights; for profiling without a checkpoint.
};

/// \brief Returns the CLI spelling ("auto", "safetensors", "pt", ...).
const char* ToString(LoadFormat format);

/// \brief Parses a CLI spelling into `*out`; InvalidArgument if unknown.
Status ParseLoadFormat(const std::string& text, LoadFormat* out);

/// \brief How weights are located and read from disk.
struct LoadConfig {
  LoadFormat load_format = LoadFormat::kAuto;
  /// Cache directory for downloaded checkpoints. Empty means the default
  /// (e.g. the Hugging Face cache).
  std::string download_dir;
  /// Glob patterns of checkpoint files to skip (vLLM: --ignore-patterns).
  std::vector<std::string> ignore_patterns;
  /// Show a progress bar while loading weights.
  bool use_progress_bar = true;
  /// Device on which `.pt` files are mapped before the copy to the GPU.
  std::string pt_load_map_location = "cpu";

  Status Validate() const;
};

}  // namespace inferx

#endif  // INFERX_CONFIG_LOAD_CONFIG_H_