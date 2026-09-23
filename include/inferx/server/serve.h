#ifndef INFERX_SERVER_SERVE_H_
#define INFERX_SERVER_SERVE_H_

#include <string>

#include "inferx/cache/cache_config.h"
#include "inferx/core/status.h"
#include "inferx/engine/execution_config.h"
#include "inferx/engine/scheduler.h"
#include "inferx/models/model_config.h"
#include "inferx/sampling/sampling_params.h"
#include "inferx/server/tokenizer_pool.h"

namespace inferx::server {

/// \brief Inputs for the OpenAI-compatible HTTP server (`inferx serve`).
///
/// The focused config groups pass straight through to the engine:
/// ModelConfig (checkpoint identity), CacheConfig (KV pool),
/// SchedulerConfig (batching), ExecutionConfig (CUDA graphs, attention
/// backend). TokenizerPoolConfig governs the prompt-preparation worker
/// pool; its tokenizer_path is filled from the model dirs by RunServe.
struct ServeParams {
  ModelConfig model;
  CacheConfig cache;
  SchedulerConfig scheduler;
  ExecutionConfig execution;
  sampling::SamplingParams default_sampling;  ///< Defaults for requests that omit them.
  TokenizerPoolConfig tokenizer;              ///< Prompt-preparation worker pool.
  std::string host = "127.0.0.1";
  int port = 8000;
  std::string served_model_name;  ///< Model name reported by the API.
};

/// \brief Runs the server until shutdown.
Status RunServe(const ServeParams& params);

}  // namespace inferx::server

#endif  // INFERX_SERVER_SERVE_H_
