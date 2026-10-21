// OpenAI endpoint handlers (see openai_handlers.h). All handler code runs
// inside the session coroutine; writes go through the exchange's
// ResponseWriter, whose `false` returns mean the peer is gone — engine work
// is cancelled via AbortGuard and nothing further is written.
#include "inferx/server/openai_handlers.h"

#include <chrono>
#include <cstdlib>
#include <ctime>
#include <optional>
#include <string>
#include <utility>

#include "inferx/server/api.h"
#include "inferx/server/api_error.h"
#include "inferx/server/request_model.h"

namespace inferx {
namespace {

/// Constant-time equality over possibly different-length strings; the
/// length mismatch itself folds into the same accumulator so timing does
/// not disclose how much matched (docs §10.5; keys are compared in memory —
/// hashing at rest is deferred until an OpenSSL dependency exists).
bool ConstantTimeEqual(const std::string& a, const std::string& b) {
  const std::size_t n = a.size() > b.size() ? a.size() : b.size();
  unsigned char diff = a.size() == b.size() ? 0 : 1;
  for (std::size_t i = 0; i < n; ++i) {
    const unsigned char x = i < a.size() ? static_cast<unsigned char>(a[i]) : 0;
    const unsigned char y = i < b.size() ? static_cast<unsigned char>(b[i]) : 0;
    diff |= static_cast<unsigned char>(x ^ y);
  }
  return diff == 0;
}

/// X-Request-Id is echoed only when it matches [A-Za-z0-9._:-]{1,128}
/// (docs §10.6); otherwise the server generates one.
std::string ResolveRequestId(const HttpExchange& exchange) {
  auto it = exchange.request.find("X-Request-Id");
  if (it != exchange.request.end()) {
    const std::string value(it->value());
    bool valid = !value.empty() && value.size() <= 128;
    for (const char c : value) {
      const bool ok = (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') ||
                      (c >= '0' && c <= '9') || c == '.' || c == '_' || c == ':' || c == '-';
      valid = valid && ok;
      if (!valid) break;
    }
    if (valid) return value;
  }
  return NewRequestId("req-");
}

/// RAII engine abort: releases the engine request exactly once unless the
/// response completed normally (mirrors vLLM's generate() cancellation).
struct AbortGuard {
  IEngineClient* client;
  MetricsService* metrics;
  std::string request_id;
  bool active = true;
  void release() { active = false; }
  ~AbortGuard() {
    if (active && client != nullptr) {
      client->Abort(request_id);
      metrics->ObserveEngineAbort();
    }
  }
};

/// True when the request carries an application/json (or omitted) content
/// type; JSON routes reject everything else with 415 (docs §8.1).
bool ContentTypeAccepted(const HttpExchange& exchange) {
  auto it = exchange.request.find(http::field::content_type);
  if (it == exchange.request.end()) return true;
  const std::string value(it->value());
  return value.rfind("application/json", 0) == 0;
}

std::string SseFrame(const Json& payload) {
  return "data: " + payload.dump() + "\n\n";
}

/// Header lists are built through a named helper: GCC 13 ICEs on
/// nested-brace temporaries passed as arguments inside co_await call
/// expressions.
std::vector<ResponseWriter::Header> AllowHeaders(const char* method,
                                                 const std::string& request_id = {}) {
  std::vector<ResponseWriter::Header> headers;
  headers.emplace_back("Allow", method);
  if (!request_id.empty()) {
    headers.emplace_back("X-Request-Id", request_id);
  }
  return headers;
}

Json MethodNotAllowed(const std::string& allow) {
  return MakeError(405, "method not allowed; allowed: " + allow, "", "method_not_allowed");
}

/// Streaming tail writer: the finish reason rides the final content chunk
/// (OpenAI semantics), a usage-only chunk follows only when requested, then
/// exactly one [DONE]. Functor, not lambda: lambdas cannot be coroutines
/// portably.
struct FinishWriter {
  ResponseWriter* writer;
  const std::string* request_id;
  const std::string* model;
  std::int64_t created;
  std::string* held_delta;
  bool include_usage;
  bool chat = false;

  net::awaitable<bool> operator()(std::string_view finish_reason, int prompt_tokens,
                                  int completion_tokens) {
    if (!co_await writer->WriteRaw(
            MakeSseChunk(*request_id, *model, created, *held_delta, finish_reason, chat))) {
      co_return false;
    }
    held_delta->clear();
    if (include_usage &&
        !co_await writer->WriteRaw(MakeSseUsageChunk(*request_id, *model, created,
                                                     prompt_tokens, completion_tokens, chat))) {
      co_return false;
    }
    if (!co_await writer->WriteDone()) co_return false;
    co_return true;
  }
};

/// One JSON error data-frame, then [DONE] (vLLM stream-error behavior).
struct StreamErrorWriter {
  ResponseWriter* writer;

  net::awaitable<bool> operator()(const ApiError& error) {
    if (!co_await writer->WriteRaw(SseFrame(MakeStreamErrorFrame(error)))) {
      co_return false;
    }
    co_return co_await writer->WriteDone();
  }
};

}  // namespace

OpenAiHandler::OpenAiHandler(HandlerConfig config, PromptProcessorPool& pool,
                             IEngineClient& engine, MetricsService& metrics,
                             std::function<bool()> draining,
                             std::function<bool()> engine_ready)
    : config_(std::move(config)),
      pool_(pool),
      engine_(engine),
      metrics_(metrics),
      draining_(std::move(draining)),
      engine_ready_(std::move(engine_ready)) {}

bool OpenAiHandler::ReadyToServe() const {
  if (draining_()) return false;
  const PoolHealth health = pool_.Health();
  if (health.unready || health.workers_down > 0 || health.workers_ready == 0) return false;
  return engine_ready_();
}

bool OpenAiHandler::AuthAllowed(const HttpExchange& exchange) const {
  if (config_.api_keys.empty()) return true;
  auto it = exchange.request.find(http::field::authorization);
  if (it == exchange.request.end()) return false;
  const std::string value(it->value());
  if (value.rfind("Bearer ", 0) != 0) return false;
  const std::string presented = value.substr(7);
  for (const std::string& key : config_.api_keys) {
    if (ConstantTimeEqual(presented, key)) return true;
  }
  return false;
}

net::awaitable<bool> OpenAiHandler::WriteApiError(HttpExchange& exchange,
                                                  const ApiError& error) {
  metrics_.ObserveRequest(exchange.path, exchange.method, error.http_status);
  std::vector<ResponseWriter::Header> headers;
  if (error.retry_after > std::chrono::seconds(0)) {
    headers.emplace_back("Retry-After", std::to_string(error.retry_after.count()));
  }
  if (error.http_status == 401) {
    headers.emplace_back("WWW-Authenticate", "Bearer");
  }
  if (exchange.path.rfind("/v1/", 0) == 0) {
    headers.emplace_back("X-Request-Id", ResolveRequestId(exchange));
  }
  co_return co_await exchange.writer.WriteJson(error.http_status, error.ToJson(),
                                               std::move(headers));
}

net::awaitable<bool> OpenAiHandler::Handle(HttpExchange& exchange) {
  const std::string& path = exchange.path;

  // Health family: public (no auth), engine-aware (docs §8.2).
  if (path == "/health") {
    if (exchange.method != "GET") {
      co_return co_await exchange.writer.WriteJson(405, MethodNotAllowed("GET"),
                                                   AllowHeaders("GET"));
    }
    // vLLM-compatible: empty 200/503 reflecting engine readiness.
    const bool ready = ReadyToServe();
    metrics_.ObserveRequest(path, exchange.method, ready ? 200 : 503);
    co_return co_await exchange.writer.WriteEmpty(ready ? 200 : 503);
  }
  if (path == "/livez") {
    if (exchange.method != "GET") {
      co_return co_await exchange.writer.WriteJson(405, MethodNotAllowed("GET"),
                                                   AllowHeaders("GET"));
    }
    metrics_.ObserveRequest(path, exchange.method, 200);
    Json body = Json::object();
    body["status"] = "ok";
    co_return co_await exchange.writer.WriteJson(200, body);
  }
  if (path == "/readyz") {
    if (exchange.method != "GET") {
      co_return co_await exchange.writer.WriteJson(405, MethodNotAllowed("GET"),
                                                   AllowHeaders("GET"));
    }
    Json body = Json::object();
    if (ReadyToServe()) {
      body["status"] = "ok";
      metrics_.ObserveRequest(path, exchange.method, 200);
      co_return co_await exchange.writer.WriteJson(200, body);
    }
    body["status"] = "unavailable";
    body["reason"] = draining_() ? "draining"
                                 : (engine_ready_() ? "prompt pool not ready" : "engine not ready");
    metrics_.ObserveRequest(path, exchange.method, 503);
    co_return co_await exchange.writer.WriteJson(503, body);
  }
  if (path == "/version") {
    if (exchange.method != "GET") {
      co_return co_await exchange.writer.WriteJson(405, MethodNotAllowed("GET"),
                                                   AllowHeaders("GET"));
    }
    Json body = Json::object();
    body["version"] = "dev";
    metrics_.ObserveRequest(path, exchange.method, 200);
    co_return co_await exchange.writer.WriteJson(200, body);
  }
  if (path == "/v1/models") {
    if (exchange.method != "GET") {
      co_return co_await exchange.writer.WriteJson(405, MethodNotAllowed("GET"),
                                                   AllowHeaders("GET"));
    }
    if (!AuthAllowed(exchange)) {
      co_return co_await WriteApiError(exchange, Unauthorized("missing or invalid API key"));
    }
    metrics_.ObserveRequest(path, exchange.method, 200);
    co_return co_await exchange.writer.WriteJson(
        200, MakeModelList(config_.model_name, config_.created));
  }
  if (path == "/metrics") {
    if (exchange.method != "GET") {
      co_return co_await exchange.writer.WriteJson(405, MethodNotAllowed("GET"),
                                                   AllowHeaders("GET"));
    }
    if (!AuthAllowed(exchange)) {
      co_return co_await WriteApiError(exchange, Unauthorized("missing or invalid API key"));
    }
    metrics_.ObservePoolHealth(pool_.Health());
    const std::string text = metrics_.Render();
    metrics_.ObserveRequest(path, exchange.method, 200);
    std::vector<ResponseWriter::Header> text_headers;
    text_headers.emplace_back("Content-Type", "text/plain; version=0.0.4; charset=utf-8");
    co_return co_await exchange.writer.WriteJson(200, text, std::move(text_headers));
  }
  if (path == "/v1/completions") {
    if (exchange.method != "POST") {
      co_return co_await exchange.writer.WriteJson(
          405, MethodNotAllowed("POST"),
          AllowHeaders("POST", ResolveRequestId(exchange)));
    }
    if (!AuthAllowed(exchange)) {
      co_return co_await WriteApiError(exchange, Unauthorized("missing or invalid API key"));
    }
    if (!ContentTypeAccepted(exchange)) {
      co_return co_await WriteApiError(exchange, UnsupportedMediaType());
    }
    co_return co_await HandleCompletions(exchange);
  }
  if (path == "/v1/chat/completions") {
    if (exchange.method != "POST") {
      co_return co_await exchange.writer.WriteJson(
          405, MethodNotAllowed("POST"),
          AllowHeaders("POST", ResolveRequestId(exchange)));
    }
    if (!AuthAllowed(exchange)) {
      co_return co_await WriteApiError(exchange, Unauthorized("missing or invalid API key"));
    }
    if (!ContentTypeAccepted(exchange)) {
      co_return co_await WriteApiError(exchange, UnsupportedMediaType());
    }
    co_return co_await HandleChatCompletions(exchange);
  }

  co_return false;  // Unknown route: the session writes the 404 envelope.
}

net::awaitable<bool> OpenAiHandler::HandleChatCompletions(HttpExchange& exchange) {
  co_return co_await HandleCompletions(exchange, true);
}

net::awaitable<bool> OpenAiHandler::HandleCompletions(HttpExchange& exchange, bool chat) {
  const ParseResult parsed =
      (chat ? ParseChatRequest : ParseCompletionsRequest)(exchange.request.body(), config_.default_sampling,
                              config_.model_name);
  if (!parsed.ok) {
    const ApiError error =
        parsed.code == "model_not_found"
            ? ModelNotFound(parsed.message)
            : ApiError{parsed.http_status,
                       parsed.http_status == 404 ? "invalid_request_error"
                                                 : "invalid_request_error",
                       parsed.code.empty() ? "invalid_request" : parsed.code,
                       parsed.message, parsed.param, {}, false};
    co_return co_await WriteApiError(exchange, error);
  }

  // Prompt preparation on the bounded worker pool; never on this I/O
  // coroutine's thread.
  const std::uint64_t prep_id = next_prep_id_.fetch_add(1);
  PromptInput input = parsed.request.token_ids.empty()
                          ? PromptInput{TextPrompt{parsed.request.prompt}}
                          : PromptInput{TokenIdsPrompt{parsed.request.token_ids}};
  if (parsed.request.chat) input = *parsed.request.chat;
  PrepareContext context;
  context.request_id = prep_id;
  StatusOr<PreparedPrompt> encoded =
      co_await pool_.Prepare(prep_id, std::move(input), context);
  if (!encoded.ok()) {
    if (encoded.status().code() == absl::StatusCode::kResourceExhausted) {
      metrics_.ObservePromptRejected("queue_full");
    } else if (encoded.status().code() == absl::StatusCode::kInvalidArgument) {
      metrics_.ObservePromptRejected("invalid");
    }
    co_return co_await WriteApiError(exchange, MapStatusToApiError(encoded.status()));
  }

  EngineCompletion completion;
  completion.prompt = std::move(encoded->token_ids);
  completion.params = parsed.request.params;
  StatusOr<EngineSubmission> submitted = engine_.Submit(std::move(completion));
  if (!submitted.ok()) {
    metrics_.ObservePromptRejected("engine_rejected");
    co_return co_await WriteApiError(exchange, MapStatusToApiError(submitted.status()));
  }

  AbortGuard guard{&engine_, &metrics_, submitted->request_id};

  const std::string request_id = NewRequestId("cmpl-").substr(5);  // prefix added by builders
  const std::int64_t created = static_cast<std::int64_t>(std::time(nullptr));

  if (!parsed.request.stream) {
    // Non-streaming: accumulate deltas, emit one response.
    std::string text;
    for (;;) {
      StatusOr<EngineEvent> event = co_await submitted->events->Next();
      if (!event.ok()) {
        if (guard.active) {
          co_return co_await WriteApiError(
              exchange, Internal("the request ended before completion"));
        }
        co_return true;  // Client gone; the guard aborts the engine request.
      }
      if (event->kind == EngineEvent::Kind::kDelta) {
        text += event->text;
        continue;
      }
      if (event->kind == EngineEvent::Kind::kFinished) {
        guard.release();
        metrics_.ObserveRequest(exchange.path, exchange.method, 200);
        std::vector<ResponseWriter::Header> id_headers;
        id_headers.emplace_back("X-Request-Id", ResolveRequestId(exchange));
        co_return co_await exchange.writer.WriteJson(
            200,
            MakeCompletionResponse(request_id, config_.model_name, created, text,
                                   FinishReasonName(event->reason), event->prompt_tokens,
                                   event->completion_tokens, chat),
            std::move(id_headers));
      }
      // kError: admission errors are 503-class, others 500. The engine's
      // raw message stays out of the client envelope (sanitization rule).
      co_return co_await WriteApiError(
          exchange,
          event->admission ? ServerOverloaded(std::chrono::seconds(1)) : Internal("generation failed"));
    }
  }

  // Streaming: SSE with the vLLM-observed contract — each delta one
  // `data:` frame, the finish reason riding the final content chunk, a
  // usage-only chunk (empty choices) only when include_usage was
  // requested, errors after headers as a JSON error frame, then exactly
  // one `data: [DONE]`.
  if (!co_await exchange.writer.BeginStream()) {
    co_return true;  // Peer gone before headers; guard aborts.
  }
  if (chat && !co_await exchange.writer.WriteRaw(
          MakeSseChunk(request_id, config_.model_name, created, "", {}, true, true)))
    co_return true;
  std::string held_delta;
  FinishWriter finish{&exchange.writer, &request_id, &config_.model_name, created,
                      &held_delta, parsed.request.include_usage, chat};
  StreamErrorWriter stream_error{&exchange.writer};

  for (;;) {
    StatusOr<EngineEvent> event = co_await submitted->events->Next();
    if (!event.ok()) {
      // Stream broke (cancel/close). Flush the held delta, then, if this
      // handler still owns the request, tell the client unless it is gone.
      if (guard.active && !exchange.writer.failed()) {
        if (!held_delta.empty()) {
          co_await exchange.writer.WriteRaw(
              MakeSseChunk(request_id, config_.model_name, created, held_delta, {}, chat));
          held_delta.clear();
        }
        co_await stream_error(EngineUnavailable("the engine stream ended unexpectedly"));
      }
      co_return true;
    }
    if (event->kind == EngineEvent::Kind::kDelta) {
      if (!held_delta.empty()) {
        if (!co_await exchange.writer.WriteRaw(MakeSseChunk(
                request_id, config_.model_name, created, held_delta, {}, chat))) {
          co_return true;  // Write failed: peer gone; guard aborts.
        }
      }
      held_delta = std::move(event->text);
      continue;
    }
    if (event->kind == EngineEvent::Kind::kFinished) {
      guard.release();
      metrics_.ObserveRequest(exchange.path, exchange.method, 200);
      co_await finish(FinishReasonName(event->reason), event->prompt_tokens,
                      event->completion_tokens);
      co_return true;
    }
    // kError after headers: flush the held delta, one JSON error frame,
    // then [DONE].
    guard.release();
    metrics_.ObserveRequest(exchange.path, exchange.method, event->admission ? 503 : 500);
    if (!held_delta.empty()) {
      if (!co_await exchange.writer.WriteRaw(
              MakeSseChunk(request_id, config_.model_name, created, held_delta, {}, chat))) {
        co_return true;
      }
      held_delta.clear();
    }
    co_await stream_error(event->admission ? ServerOverloaded(std::chrono::seconds(1))
                                           : Internal("generation failed"));
    co_return true;
  }
}

}  // namespace inferx
