#ifndef INFERX_SERVER_SERVE_H_
#define INFERX_SERVER_SERVE_H_

#include "inferx/config/cache_config.h"
#include "inferx/config/execution_config.h"
#include "inferx/config/model_config.h"
#include "inferx/config/parallel_config.h"
#include "inferx/config/scheduler_config.h"
#include "inferx/config/server_config.h"
#include "inferx/core/status.h"
#include "inferx/sampling/sampling_params.h"

namespace inferx {

/// \brief Everything `RunServe` needs, grouped by owning subsystem: the
///        serving frontend's tunables, the engine's five configs, and the
///        request sampling defaults.
struct ServeConfig {
  /// Frontend: listen address, auth, executors, limits, and timeouts.
  ServerConfig server;

  /// Engine: checkpoint directory, device placement, tokenizer resolution.
  ModelConfig model;

  /// Engine: KV cache sizing.
  CacheConfig cache;

  /// Engine: admission and batching (queue capacity, max sequences).
  SchedulerConfig scheduler;

  /// Engine: CUDA graph replay.
  ExecutionConfig execution;

  /// Engine: tensor-parallel topology.
  ParallelConfig parallel;

  /// Request sampling defaults: the user's explicit flags merged over the
  /// checkpoint's generation config (MergeGenerationConfig).
  sampling::SamplingParams default_sampling;

  /// \brief Validates every member that has a validator.
  ///
  /// \return OK, or the first error (the offending field named in the
  ///         message).
  [[nodiscard]] Status Validate() const;
};

/// \brief Runs the OpenAI-compatible HTTP server until shutdown.
///
/// This is the composition root of the serving stack. Validates the
/// configuration, then builds the stack in construction order — model
/// loading (checkpoint weights onto the device, memory profiling, the KV
/// cache pool, warm-up), the engine client over the scheduler, and the
/// HTTP frontend — serves until a shutdown signal drains the frontend,
/// and tears down in reverse: bounded frontend stop, then engine
/// shutdown.
///
/// \param config Resolved serving configuration (already environment-merged).
/// \return       OK after a complete drain and engine shutdown, or the
///               first configuration or fatal error. An error from the
///               final engine shutdown means the engine must not be
///               destroyed; the supervisor terminates the process.
Status RunServe(const ServeConfig& config);

}  // namespace inferx

#endif  // INFERX_SERVER_SERVE_H_
