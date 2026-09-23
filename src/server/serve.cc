// `inferx serve` assembly (server.md §4): coroutine HTTP front end over
// the EngineGateway. This file owns API semantics (dispatch, the
// completions request path, SSE writing); http_server.cc owns connection
// plumbing and engine_gateway.cc owns the engine thread.
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <ctime>
#include <memory>
#include <optional>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "inferx/core/status_util.h"
#include "inferx/server/api.h"
#include "inferx/server/engine_gateway.h"
#include "inferx/server/http_server.h"
#include "inferx/server/serve.h"
#include "inferx/tokenizer/tokenizer.h"

namespace inferx::server {
namespace {

constexpr std::chrono::seconds kShutdownGrace{10};

/// Aborts the engine request if the coroutine exits for any reason
/// (response complete, client disconnect, write error).
struct CancelGuard {
  EngineGateway& gateway;
  std::uint64_t id;
  ~CancelGuard() { gateway.Cancel(id); }
};

net::awaitable<void> WriteJsonResponse(
    beast::tcp_stream& stream, const http::request<http::string_body>& req,
    int status, const Json& body, const char* retry_after = nullptr) {
  http::response<http::string_body> res{
      static_cast<http::status>(status), req.version()};
  res.set(http::field::server, "inferx");
  res.set(http::field::content_type, "application/json");
  if (retry_after != nullptr) res.set("Retry-After", retry_after);
  res.keep_alive(req.keep_alive());
  res.body() = body.dump();
  res.prepare_payload();
  co_await http::async_write(stream, res, net::use_awaitable);
}

net::awaitable<void> WriteSseHeader(beast::tcp_stream& stream,
                                    const http::request<http::string_body>& req) {
  http::response<http::empty_body> res{http::status::ok, req.version()};
  res.set(http::field::server, "inferx");
  res.set(http::field::content_type, "text/event-stream");
  res.set(http::field::cache_control, "no-cache");
  res.keep_alive(req.keep_alive());
  res.chunked(true);
  http::response_serializer<http::empty_body> serializer{res};
  co_await http::async_write_header(stream, serializer, net::use_awaitable);
}

/// One HTTP chunk carrying `payload`; flushed immediately so per-event
/// timing stays honest.
net::awaitable<void> WriteChunk(beast::tcp_stream& stream, std::string payload) {
  if (payload.empty()) co_return;
  char prefix[32];
  const int digits = std::snprintf(prefix, sizeof(prefix), "%zX\r\n", payload.size());
  std::string frame;
  frame.reserve(payload.size() + digits + 2);
  frame.append(prefix, digits);
  frame += payload;
  frame += "\r\n";
  co_await net::async_write(stream, net::buffer(frame), net::use_awaitable);
}

net::awaitable<void> FinishChunkedBody(beast::tcp_stream& stream) {
  co_await net::async_write(stream, net::buffer("0\r\n\r\n", 5), net::use_awaitable);
}

/// Per-request streaming state; the held-back delta lets the finish reason
/// ride on the last content chunk, exactly like OpenAI streams.
struct StreamState {
  bool header_written = false;
  std::string held_delta;
};

class InferxDispatcher : public RequestDispatcher {
 public:
  InferxDispatcher(const ServeParams& params, EngineGateway& gateway)
      : model_(params.served_model_name.empty() ? params.model.model_dir
                                                : params.served_model_name),
        defaults_(params.default_sampling),
        gateway_(gateway),
        created_(static_cast<std::int64_t>(std::time(nullptr))) {}

  bool HandleAdmin(const std::string& method, const std::string& target, int& status,
                   std::string& content_type, std::string& body) override {
    if (target == "/health" && method == "GET") {
      status = 200;
      content_type = "application/json";
      body = "{\"status\":\"ok\"}";
      return true;
    }
    if (target == "/v1/models" && method == "GET") {
      status = 200;
      content_type = "application/json";
      body = MakeModelList(model_, created_).dump();
      return true;
    }
    return false;
  }

  net::awaitable<void> HandleCompletions(
      beast::tcp_stream& stream, const http::request<http::string_body>& req) override {
    const ParseResult parsed =
        ParseCompletionsRequest(req.body(), defaults_, model_);
    if (!parsed.ok) {
      co_await WriteJsonResponse(stream, req, parsed.http_status,
                                 MakeError(parsed.http_status, parsed.message,
                                           parsed.param, parsed.code));
      co_return;
    }
    StatusOr<std::vector<int>> encoded = gateway_.Encode(parsed.request.prompt);
    if (!encoded.ok()) {
      co_await WriteJsonResponse(stream, req, 400,
                                 MakeError(400, "failed to tokenize prompt", "prompt"));
      co_return;
    }
    StatusOr<SubmitResult> submit =
        gateway_.Submit({std::move(*encoded), parsed.request.params});
    if (!submit.ok()) {
      co_await WriteJsonResponse(
          stream, req, 503,
          MakeError(503, std::string(submit.status().message()), "", "overloaded"),
          "1");
      co_return;
    }
    CancelGuard guard{gateway_, submit->id};
    if (parsed.request.stream) {
      co_await RunStreaming(stream, req, *submit);
    } else {
      co_await RunNonStreaming(stream, req, *submit);
    }
  }

 private:
  net::awaitable<void> RunNonStreaming(
      beast::tcp_stream& stream, const http::request<http::string_body>& req,
      const SubmitResult& submit) {
    std::string text;
    for (;;) {
      auto batch = co_await submit.events->TakeAll();
      if (!batch.has_value()) {
        co_await WriteJsonResponse(stream, req, 500,
                                   MakeError(500, "request cancelled"));
        co_return;
      }
      for (const CompletionEvent& event : *batch) {
        if (event.kind == CompletionEvent::Kind::kDelta) {
          text += event.delta;
          continue;
        }
        if (event.kind == CompletionEvent::Kind::kError) {
          co_await WriteJsonResponse(
              stream, req, event.admission ? 503 : 500,
              MakeError(event.admission ? 503 : 500, event.message, "",
                        event.admission ? "overloaded" : ""),
              event.admission ? "1" : nullptr);
          co_return;
        }
        co_await WriteJsonResponse(
            stream, req, 200,
            MakeCompletionResponse(std::to_string(submit.id), model_, created_, text,
                                   FinishReasonName(event.reason),
                                   event.prompt_tokens, event.completion_tokens));
        co_return;
      }
    }
  }

  net::awaitable<void> RunStreaming(beast::tcp_stream& stream,
                                    const http::request<http::string_body>& req,
                                    const SubmitResult& submit) {
    StreamState state;
    for (;;) {
      auto batch = co_await submit.events->TakeAll();
      if (!batch.has_value()) co_return;  // Cancelled; teardown closes us.
      // One write per batch: syscalls scale with engine steps, not tokens.
      for (const CompletionEvent& event : *batch) {
        if (co_await WriteStreamEvent(stream, req, submit, state, event)) co_return;
      }
    }
  }

  /// \brief Writes one event. Returns true when the response is complete.
  /// Callers hand TakeAll() batches here, so each engine step costs one
  /// write. The finish reason rides on the last content chunk.
  net::awaitable<bool> WriteStreamEvent(beast::tcp_stream& stream,
                                        const http::request<http::string_body>& req,
                                        const SubmitResult& submit, StreamState& state,
                                        const CompletionEvent& event) {
    const std::string id = std::to_string(submit.id);
    auto ensure_header = [&]() -> net::awaitable<void> {
      if (state.header_written) co_return;
      co_await WriteSseHeader(stream, req);
      state.header_written = true;
    };

    if (event.kind == CompletionEvent::Kind::kDelta) {
      co_await ensure_header();
      std::string payload = MakeSseChunk(id, model_, created_, state.held_delta, {});
      state.held_delta = event.delta;
      co_await WriteChunk(stream, std::move(payload));
      co_return false;
    }
    if (event.kind == CompletionEvent::Kind::kFinish) {
      co_await ensure_header();
      std::string payload =
          MakeSseChunk(id, model_, created_, state.held_delta,
                       FinishReasonName(event.reason));
      state.held_delta.clear();
      payload += MakeSseUsageChunk(id, model_, created_, event.prompt_tokens,
                                   event.completion_tokens);
      payload += kSseDone;
      co_await WriteChunk(stream, std::move(payload));
      co_await FinishChunkedBody(stream);
      co_return true;
    }
    // kError: without a streamed byte yet the response is still a clean
    // HTTP error; otherwise terminate the stream best-effort.
    if (!state.header_written) {
      co_await WriteJsonResponse(
          stream, req, event.admission ? 503 : 500,
          MakeError(event.admission ? 503 : 500, event.message, "",
                    event.admission ? "overloaded" : ""),
          event.admission ? "1" : nullptr);
      co_return true;
    }
    std::string payload = MakeSseChunk(id, model_, created_, state.held_delta, {});
    state.held_delta.clear();
    payload += kSseDone;
    co_await WriteChunk(stream, std::move(payload));
    co_await FinishChunkedBody(stream);
    co_return true;
  }

  std::string model_;
  sampling::SamplingParams defaults_;
  EngineGateway& gateway_;
  std::int64_t created_;
};

}  // namespace

Status RunServe(const ServeParams& params) {
  return Guarded([&]() -> Status {
    auto tokenizer = Take(Tokenizer::FromFile(
        (params.model.tokenizer_dir.empty() ? params.model.model_dir
                                            : params.model.tokenizer_dir) +
        "/tokenizer.json"));

    net::io_context io;
    EngineGateway gateway(io, params.model, params.cache, params.scheduler,
                          params.execution, tokenizer);
    InferxDispatcher dispatcher(params, gateway);
    HttpServer server(io, params.host, static_cast<std::uint16_t>(params.port),
                      dispatcher);
    server.Listen();

    // SIGINT/SIGTERM: stop accepting, let the engine drain live requests,
    // force-stop after a grace period.
    net::signal_set signals(io, SIGINT, SIGTERM);
    auto grace = std::make_shared<net::steady_timer>(io);
    signals.async_wait([&server, &gateway, &io, grace](boost::system::error_code ec,
                                                       int) {
      if (ec) return;
      server.Stop();
      gateway.Shutdown();  // Drains live requests, then stops the engine.
      grace->expires_after(kShutdownGrace);
      grace->async_wait([&io, grace](boost::system::error_code timer_ec) {
        if (!timer_ec) io.stop();
      });
    });

    const unsigned hardware = std::max(1u, std::thread::hardware_concurrency());
    const unsigned threads_count = std::clamp(hardware, 2u, 4u);
    std::vector<std::thread> workers;
    workers.reserve(threads_count - 1);
    for (unsigned i = 1; i < threads_count; ++i) {
      workers.emplace_back([&io] { io.run(); });
    }
    io.run();
    for (std::thread& worker : workers) worker.join();
    gateway.Shutdown();  // Idempotent; joins the engine thread.
    return OkStatus();
  });
}

}  // namespace inferx::server
