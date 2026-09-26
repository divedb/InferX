#ifndef INFERX_SERVER_SERVE_H_
#define INFERX_SERVER_SERVE_H_

#include <set>
#include <string>

#include "inferx/config/cache_config.h"
#include "inferx/core/status.h"
#include "inferx/config/execution_config.h"
#include "inferx/config/scheduler_config.h"
#include "inferx/config/model_config.h"
#include "inferx/models/checkpoint_config.h"
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
};

/// \brief Merges checkpoint generation defaults into `sampling`.
///
/// A field named in `skip` keeps the caller's value -- used for fields the
/// user set explicitly, which must win over the checkpoint file. HF top_k
/// <= 0 means "disabled" and maps to 0.
void MergeGenerationConfig(const GenerationConfig& gen,
                           sampling::SamplingParams* sampling,
                           const std::set<std::string>& skip = {});

/// \brief Runs the server until shutdown.
Status RunServe(const ServeParams& params);

}  // namespace inferx::server

#endif  // INFERX_SERVER_SERVE_H_
