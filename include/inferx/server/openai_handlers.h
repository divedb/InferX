/// \brief OpenAI-compatible endpoint handlers (docs/inference_server_design.md
/// §8, reconciled to InferX).
///
/// Owns routing decisions, auth, validation, normalization, prompt
/// preparation, engine submission, and response/SSE writing for the
/// endpoint matrix of §3.4: /health, /livez, /readyz, /version, /v1/models,
/// /metrics, /v1/completions, /v1/chat/completions. Engine interaction is
/// exclusively through IEngineClient; prompt work through the
/// PromptProcessorPool. Depends on no engine headers at link time, so the
/// whole surface is testable with a fake engine.
#ifndef INFERX_SERVER_OPENAI_HANDLERS_H_
#define INFERX_SERVER_OPENAI_HANDLERS_H_

#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "inferx/sampling/sampling_params.h"
#include "inferx/server/api_error.h"
#include "inferx/server/engine_client.h"
#include "inferx/server/http_server.h"
#include "inferx/server/metrics.h"
#include "inferx/server/prompt_processor_pool.h"

namespace inferx {

struct HandlerConfig {
  std::string model_name;              ///< Served model name (vLLM served-model-name).
  std::int64_t created = 0;            ///< Stable `created` epoch seconds.
  sampling::SamplingParams default_sampling;
  /// Configured bearer keys; empty disables auth (vLLM --api-keys).
  std::vector<std::string> api_keys;
};

class OpenAiHandler : public RequestHandler {
 public:
  /// `draining`/`engine_ready` are cheap snapshots read per request.
  OpenAiHandler(HandlerConfig config, PromptProcessorPool& pool, IEngineClient& engine,
                MetricsService& metrics, std::function<bool()> draining,
                std::function<bool()> engine_ready);

  net::awaitable<bool> Handle(HttpExchange& exchange) override;

 private:
  net::awaitable<bool> HandleCompletions(HttpExchange& exchange, bool chat = false);
  net::awaitable<bool> HandleChatCompletions(HttpExchange& exchange);
  net::awaitable<bool> WriteApiError(HttpExchange& exchange, const ApiError& error);

  /// 401 unless a configured bearer key matches; empty key set passes.
  bool AuthAllowed(const HttpExchange& exchange) const;

  bool ReadyToServe() const;

  HandlerConfig config_;
  PromptProcessorPool& pool_;
  IEngineClient& engine_;
  MetricsService& metrics_;
  std::function<bool()> draining_;
  std::function<bool()> engine_ready_;
  std::atomic<std::uint64_t> next_prep_id_{1};
};

}  // namespace inferx

#endif  // INFERX_SERVER_OPENAI_HANDLERS_H_
