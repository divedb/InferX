#ifndef INFERX_MODELS_LOAD_OPTIONS_H_
#define INFERX_MODELS_LOAD_OPTIONS_H_

#include <string>

#include "inferx/config/parallel_config.h"
#include "inferx/core/device.h"

namespace inferx::models {

/// \brief Checkpoint location and loading policy for one model load.
///
/// Owns everything the loader needs to decide *how* to load: where the
/// checkpoint is, what dtype execution requested, and the tensor-parallel
/// topology this rank belongs to. Architecture dimensions come from the
/// checkpoint's own configuration, not from here. `max_tokens` and
/// `max_seqs` are the execution capacities workspace is preallocated for;
/// graph-captured forwards require the addresses that follow from them to
/// stay stable.
struct ModelLoadOptions {
  /// Directory holding `config.json` and `.safetensors` weights.
  std::string directory;

  /// Device the weights and workspace are placed on.
  DeviceId device;

  /// Requested execution dtype; "auto" follows the checkpoint. Unsupported
  /// dtypes fail before weight allocation.
  std::string dtype = "auto";

  /// Workspace capacity for one step's token batch.
  int max_tokens = 0;

  /// Workspace capacity for concurrently sampled sequences.
  int max_seqs = 0;

  /// Tensor-parallel topology of this rank; defaults shard nothing.
  ParallelConfig parallel;
};

}  // namespace inferx::models

#endif  // INFERX_MODELS_LOAD_OPTIONS_H_
