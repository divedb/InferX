#ifndef INFERX_SERVER_SERVE_H_
#define INFERX_SERVER_SERVE_H_

#include <memory>
#include <set>
#include <string>
#include <vector>

#include "inferx/config/cache_config.h"
#include "inferx/config/execution_config.h"
#include "inferx/config/model_config.h"
#include "inferx/config/parallel_config.h"
#include "inferx/config/scheduler_config.h"
#include "inferx/core/status.h"
#include "inferx/sampling/sampling_params.h"
#include "inferx/server/generation_config.h"
#include "inferx/server/prompt_processor_pool.h"

namespace inferx {

/// \brief Inputs for the OpenAI-compatible HTTP server (`inferx serve`).
///
/// The focused config groups pass straight through to the engine:
/// ModelConfig (checkpoint identity), CacheConfig (KV pool),
/// SchedulerConfig (batching), ExecutionConfig (CUDA graphs, attention
/// backend). PromptPoolConfig governs the prompt-preparation worker
/// pool; its tokenizer_path is filled from the model dirs by RunServe.
struct EngineConfig {
  ModelConfig model;
  CacheConfig cache;
  SchedulerConfig scheduler;
  ExecutionConfig execution;
  ParallelConfig parallel;
  Status Validate() const;
};

struct FrontendConfig {
  std::string host = "127.0.0.1";
  int port = 8000;
  PromptPoolConfig tokenizer;
  sampling::SamplingParams default_sampling;
  /// Bearer API keys guarding /v1 and /metrics (vLLM --api-keys); empty
  /// disables auth. INFERX_API_KEY is merged in at startup.
  std::vector<std::string> api_keys;
};

struct ServeParams {
  EngineConfig engine;
  FrontendConfig frontend;
};

/// \brief Merges checkpoint generation defaults into `sampling`.
///
/// A field named in `skip` keeps the caller's value -- used for fields the
/// user set explicitly, which must win over the checkpoint file. HF top_k
/// <= 0 means "disabled" and maps to 0.
void MergeGenerationConfig(const GenerationConfig& gen, sampling::SamplingParams* sampling,
                           const std::set<std::string>& skip = {});

/// \brief Runs the server until shutdown.
Status RunServe(const ServeParams& params);

}  // namespace inferx

#endif  // INFERX_SERVER_SERVE_H_
