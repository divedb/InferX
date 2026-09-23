/// \file
/// \brief User-facing model configuration (vLLM ModelConfig analogue).

#ifndef INFERX_MODEL_CONFIG_H_
#define INFERX_MODEL_CONFIG_H_

#include <cstdint>
#include <string>

#include "inferx/core/device.h"

namespace inferx {

/// \brief Which checkpoint to load and how the engine executes it.
///
/// Focused counterpart to the removed ModelRunnerConfig monolith: the
/// dimensions parsed from the checkpoint live in CheckpointConfig, KV pool
/// sizing in CacheConfig, batch capacity in SchedulerConfig, and CUDA graph
/// / attention backend selection in ExecutionConfig. Fields the runtime has
/// not adopted yet (dtype overrides, seed, max_model_len) are carried here
/// until each feature lands.
struct ModelConfig {
  std::string model_dir;                ///< Checkpoint directory.
  /// \brief Directory holding tokenizer.json; empty derives from model_dir
  ///        (vLLM: --tokenizer).
  std::string tokenizer_dir;
  DeviceId device = DeviceId::Cuda(0);  ///< Placement device.
  /// \brief Compute dtype override: "auto" (the default) follows the
  ///        checkpoint, else bfloat16/float16/float32.
  std::string dtype = "auto";
  /// \brief Random seed for reproducible sampling.
  std::int64_t seed = 0;
  /// \brief Maximum sequence length (prompt plus output); 0 keeps the
  ///        checkpoint's context limit.
  std::int64_t max_model_len = 0;
};

}  // namespace inferx

#endif  // INFERX_MODEL_CONFIG_H_
