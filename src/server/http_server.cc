#include "inferx/server/http_server.h"

#include <atomic>
#include <chrono>
#include <cstdio>

#include "httplib.h"
#include "inferx/api/openai.h"
#include "inferx/observe/metrics.h"
#include "inferx/tokenizer/chat_template.h"

namespace inferx::server {
namespace {

using api::FinishReason;
using api::SamplingRequest;
using api::Usage;

int64_t NowSeconds() {
  return std::chrono::duration_cast<std::chrono::seconds>(
             std::chrono::system_clock::now().time_since_epoch())
      .count();
}

// Response ids are informational, so a counter is enough -- they need to be
// distinct within a process, not unguessable.
std::string MakeId(std::string_view prefix) {
  static std::atomic<uint64_t> counter{0};

  return std::string(prefix) + "-" +
         std::to_string(counter.fetch_add(1, std::memory_order_relaxed));
}

FinishReason ToApiReason(scheduler::FinishReason reason) {
  switch (reason) {
    case scheduler::FinishReason::kStopToken:
      return FinishReason::kStop;
    case scheduler::FinishReason::kMaxTokens:
    case scheduler::FinishReason::kOutOfMemory:
    case scheduler::FinishReason::kContextLimit:
    case scheduler::FinishReason::kCancelled:
    case scheduler::FinishReason::kNotFinished:
      // Everything that is not a clean stop is truncation from the client's
      // side, and `length` is the only word the API has for that.
      return FinishReason::kLength;
  }

  return FinishReason::kLength;
}


// The API's view of sampling, as the scheduler wants it. A seed the caller did
// not pin is derived per request rather than left at zero, so two concurrent
// requests at the same temperature do not draw identically.
scheduler::SamplingParams ToSchedulerParams(const SamplingRequest& s) {
  static std::atomic<uint64_t> counter{0x243f6a8885a308d3ULL};

  scheduler::SamplingParams params;
  params.temperature = s.temperature;
  params.top_p = s.top_p;
  params.seed = s.has_seed
                    ? s.seed
                    : counter.fetch_add(0x9e3779b97f4a7c15ULL,
                                        std::memory_order_relaxed);

  return params;
}

void SendError(httplib::Response* response, int code, std::string_view message,
               std::string_view type) {
  response->status = code;
  response->set_content(api::ErrorJson(message, type), "application/json");
}

// Maps a Status onto the HTTP code a client should see. InvalidArgument is the
// caller's fault; everything else is ours.
int StatusToHttp(const Status& status) {
  switch (status.code()) {
    case absl::StatusCode::kInvalidArgument:
      return 400;
    case absl::StatusCode::kUnimplemented:
      return 501;
    case absl::StatusCode::kResourceExhausted:
      return 429;
    case absl::StatusCode::kNotFound:
      return 404;
    default:
      return 500;
  }
}

std::string RenderMetrics(const Engine::Stats& stats) {
  observe::Registry registry;
  auto gauge = [&](std::string name, std::string help, double value,
                   observe::Labels labels = {}) {
    registry.AddGauge(std::move(name), std::move(help), std::move(labels))
        ->Set(value);
  };
  auto counter = [&](std::string name, std::string help, uint64_t value) {
    registry.AddCounter(std::move(name), std::move(help))->Increment(value);
  };

  gauge("inferx_requests_running", "Requests currently executing.",
        stats.running);
  gauge("inferx_requests_waiting", "Requests waiting for admission.",
        stats.waiting);
  gauge("inferx_kv_blocks", "KV-cache blocks by state.", stats.blocks_in_use,
        {{"state", "used"}});
  gauge("inferx_kv_blocks", "KV-cache blocks by state.", stats.blocks_total,
        {{"state", "total"}});
  gauge("inferx_kv_blocks", "KV-cache blocks by state.", stats.cached_blocks,
        {{"state", "cached"}});
  const uint64_t free_blocks = stats.blocks_total > stats.blocks_in_use
                                   ? stats.blocks_total - stats.blocks_in_use
                                   : 0;
  gauge("inferx_kv_blocks", "KV-cache blocks by state.", free_blocks,
        {{"state", "free"}});
  gauge("inferx_kv_cache_usage_ratio", "Fraction of KV-cache blocks in use.",
        stats.blocks_total == 0
            ? 0.0
            : static_cast<double>(stats.blocks_in_use) / stats.blocks_total);
  gauge("inferx_engine_last_step_seconds",
        "Device duration of the most recently completed engine step.",
        stats.last_step_ms / 1000.0);
  counter("inferx_steps_total", "Completed engine steps.", stats.steps);
  counter("inferx_generation_tokens_total", "Generated output tokens.",
          stats.tokens_generated);
  counter("inferx_preemptions_total", "Scheduler preemptions.",
          stats.preemptions);
  counter("inferx_prefix_cache_hits_total", "Prompt tokens served from cache.",
          stats.prefix_hit_tokens);
  counter("inferx_prefix_cache_misses_total",
          "Prompt tokens not served from cache.", stats.prefix_miss_tokens);
  counter("inferx_prefix_cache_evictions_total", "Evicted prefix-cache blocks.",
          stats.evicted_blocks);
  return registry.Render();
}

std::string RenderStatsJson(const Engine::Stats& stats) {
  return "{\"running\":" + std::to_string(stats.running) +
         ",\"waiting\":" + std::to_string(stats.waiting) +
         ",\"blocks_in_use\":" + std::to_string(stats.blocks_in_use) +
         ",\"blocks_total\":" + std::to_string(stats.blocks_total) +
         ",\"steps\":" + std::to_string(stats.steps) +
         ",\"tokens_generated\":" + std::to_string(stats.tokens_generated) +
         ",\"last_step_ms\":" + std::to_string(stats.last_step_ms) +
         ",\"preemptions\":" + std::to_string(stats.preemptions) +
         ",\"cached_blocks\":" + std::to_string(stats.cached_blocks) +
         ",\"prefix_hit_tokens\":" + std::to_string(stats.prefix_hit_tokens) +
         ",\"prefix_miss_tokens\":" + std::to_string(stats.prefix_miss_tokens) +
         ",\"evicted_blocks\":" + std::to_string(stats.evicted_blocks) + "}";
}

}  // namespace

struct HttpServer::Impl {
  Engine* engine;
  HttpServerConfig config;
  httplib::Server server;

  // The port actually bound, which differs from the configured one when that
  // was 0 and the kernel chose. Atomic because a test binds on one thread and
  // reads the port from another.
  std::atomic<int> bound_port{0};

  Impl(Engine* e, HttpServerConfig c) : engine(e), config(std::move(c)) {}

  // Runs a request to completion and returns the whole body at once.
  void ServeBlocking(const SamplingRequest& sampling,
                     std::vector<int32_t> prompt, bool chat,
                     httplib::Response* response) {
    StatusOr<std::shared_ptr<Generation>> generation = engine->Submit(
        std::move(prompt), sampling.max_tokens, sampling.stop,
        ToSchedulerParams(sampling));

    if (!generation.ok()) {
      SendError(response, StatusToHttp(generation.status()),
                generation.status().message(), "invalid_request_error");
      return;
    }

    const bool sampled = sampling.temperature > 0.0f;

    std::string text;
    Generation::Event event;
    scheduler::FinishReason reason = scheduler::FinishReason::kNotFinished;
    int32_t generated = 0;

    while ((*generation)->Next(&event)) {
      if (event.done) {
        reason = event.reason;
        generated = event.generated;
        break;
      }

      text += event.text;
      generated = event.generated;
    }

    Usage usage;
    usage.prompt_tokens = (*generation)->prompt_tokens();
    usage.completion_tokens = generated;

    const std::string id = MakeId(chat ? "chatcmpl" : "cmpl");
    const int64_t created = NowSeconds();

    response->set_content(
        chat ? api::ChatCompletionJson(id, engine->model_name(), text,
                                       ToApiReason(reason), usage, created, sampled)
             : api::CompletionJson(id, engine->model_name(), text,
                                   ToApiReason(reason), usage, created, sampled),
        "application/json");
  }

  // Streams a request as Server-Sent Events.
  void ServeStreaming(const SamplingRequest& sampling,
                      std::vector<int32_t> prompt, bool chat,
                      httplib::Response* response) {
    StatusOr<std::shared_ptr<Generation>> generation = engine->Submit(
        std::move(prompt), sampling.max_tokens, sampling.stop,
        ToSchedulerParams(sampling));

    if (!generation.ok()) {
      SendError(response, StatusToHttp(generation.status()),
                generation.status().message(), "invalid_request_error");
      return;
    }

    const bool sampled = sampling.temperature > 0.0f;
    const std::string id = MakeId(chat ? "chatcmpl" : "cmpl");
    const int64_t created = NowSeconds();
    const std::string model = engine->model_name();

    std::shared_ptr<Generation> stream = *generation;

    // Buffering would defeat the point: the client is waiting to render tokens
    // as they arrive, so the proxy hint goes out with the headers.
    response->set_header("Cache-Control", "no-cache");
    response->set_header("X-Accel-Buffering", "no");

    const bool include_usage = sampling.include_usage;

    response->set_chunked_content_provider(
        "text/event-stream",
        [this, stream, id, created, model, chat, sampled, include_usage](
            size_t /*offset*/, httplib::DataSink& sink) {
          // The first chunk announces the role and carries no content, which is
          // what OpenAI's protocol specifies and what clients key on to open
          // the message.
          if (chat) {
            const std::string first = api::SseFrame(api::ChatCompletionChunkJson(
                id, model, "assistant", "", nullptr, created, sampled));

            if (!sink.write(first.data(), first.size())) {
              stream->Cancel();
              return false;
            }
          }

          Generation::Event event;

          while (stream->Next(&event)) {
            std::string frame;

            if (event.done) {
              const FinishReason reason = ToApiReason(event.reason);

              frame = api::SseFrame(
                  chat ? api::ChatCompletionChunkJson(id, model, "", "",
                                                      &reason, created, sampled)
                       : api::CompletionChunkJson(id, model, "", &reason,
                                                  created, sampled));

              // The accounting chunk goes between the finish reason and
              // [DONE], which is where OpenAI puts it: a client that stops
              // reading at the finish reason loses only the counts, and one
              // that reads to [DONE] gets them without a second request.
              if (include_usage) {
                Usage usage;
                usage.prompt_tokens = stream->prompt_tokens();
                usage.completion_tokens = event.generated;

                frame += api::SseFrame(api::UsageChunkJson(
                    id, model, usage, created, chat, sampled));
              }

              frame += api::SseFrame("[DONE]");

              sink.write(frame.data(), frame.size());
              sink.done();

              return true;
            }

            // An empty delta is normal -- a token can complete no character --
            // and sending a chunk for it would be noise on the wire.
            if (event.text.empty()) continue;

            frame = api::SseFrame(
                chat ? api::ChatCompletionChunkJson(id, model, "", event.text,
                                                    nullptr, created, sampled)
                     : api::CompletionChunkJson(id, model, event.text, nullptr,
                                                created, sampled));

            // A failed write means the client is gone. Cancelling here is what
            // stops the engine generating into a socket nobody is reading --
            // §4's step 10, and the only backpressure the server has.
            if (!sink.write(frame.data(), frame.size())) {
              stream->Cancel();
              return false;
            }
          }

          sink.done();
          return true;
        },
        // Called when the connection ends for any reason, including one the
        // provider above never observes because it is blocked in Next.
        [stream](bool /*success*/) { stream->Cancel(); });
  }

  void Route() {
    server.Get("/health", [](const httplib::Request&,
                             httplib::Response& response) {
      response.set_content("{\"status\":\"ok\"}", "application/json");
    });

    server.Get("/v1/models", [this](const httplib::Request&,
                                    httplib::Response& response) {
      response.set_content(api::ModelsJson(engine->model_name(), NowSeconds()),
                           "application/json");
    });

    server.Get("/metrics", [this](const httplib::Request&,
                                  httplib::Response& response) {
      response.set_content(RenderMetrics(engine->stats()),
                           "text/plain; version=0.0.4; charset=utf-8");
    });

    server.Get("/stats", [this](const httplib::Request&,
                                httplib::Response& response) {
      response.set_content(RenderStatsJson(engine->stats()),
                           "application/json");
    });

    server.Post("/v1/chat/completions", [this](const httplib::Request& request,
                                               httplib::Response& response) {
      StatusOr<api::ChatCompletionRequest> parsed =
          api::ParseChatCompletionRequest(request.body);

      if (!parsed.ok()) {
        SendError(&response, StatusToHttp(parsed.status()),
                  parsed.status().message(), "invalid_request_error");
        return;
      }

      const StatusOr<std::string> prompt = tokenizer::ApplyQwen2ChatTemplate(
          parsed->messages, /*add_generation_prompt=*/true);

      if (!prompt.ok()) {
        SendError(&response, StatusToHttp(prompt.status()),
                  prompt.status().message(), "invalid_request_error");
        return;
      }

      // Encode, not EncodeOrdinary: the template's control tokens are meant as
      // control tokens. The user's own text was already escaped into the
      // template as data, and the template is the only thing that puts
      // <|im_start|> in this string.
      std::vector<int32_t> ids = engine->tokenizer().Encode(*prompt);

      if (parsed->sampling.stream) {
        ServeStreaming(parsed->sampling, std::move(ids), /*chat=*/true,
                       &response);
      } else {
        ServeBlocking(parsed->sampling, std::move(ids), /*chat=*/true,
                      &response);
      }
    });

    server.Post("/v1/completions", [this](const httplib::Request& request,
                                          httplib::Response& response) {
      StatusOr<api::CompletionRequest> parsed =
          api::ParseCompletionRequest(request.body);

      if (!parsed.ok()) {
        SendError(&response, StatusToHttp(parsed.status()),
                  parsed.status().message(), "invalid_request_error");
        return;
      }

      // A raw completion prompt is user text all the way through, so control
      // tokens in it are characters, not turn boundaries.
      std::vector<int32_t> ids =
          engine->tokenizer().EncodeOrdinary(parsed->prompt);

      if (ids.empty()) {
        SendError(&response, 400, "prompt encodes to no tokens",
                  "invalid_request_error");
        return;
      }

      if (parsed->sampling.stream) {
        ServeStreaming(parsed->sampling, std::move(ids), /*chat=*/false,
                       &response);
      } else {
        ServeBlocking(parsed->sampling, std::move(ids), /*chat=*/false,
                      &response);
      }
    });

    server.set_exception_handler([](const httplib::Request&,
                                    httplib::Response& response,
                                    std::exception_ptr /*ep*/) {
      SendError(&response, 500, "internal error", "server_error");
    });
  }
};

HttpServer::HttpServer(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}

HttpServer::~HttpServer() { Stop(); }

StatusOr<std::unique_ptr<HttpServer>> HttpServer::Create(
    Engine* engine, const HttpServerConfig& config) {
  if (engine == nullptr) return InvalidArgumentError("engine is null");

  auto impl = std::make_unique<Impl>(engine, config);

  impl->server.set_payload_max_length(config.max_request_bytes);
  impl->server.set_read_timeout(config.read_timeout_seconds, 0);

  // Nagle's algorithm and an SSE stream are a bad pair, and cpp-httplib leaves
  // TCP_NODELAY off by default. Each token is one small write; Nagle holds it
  // in the kernel until the previous small segment is acknowledged, and the
  // peer's delayed ACK does not fire for ~40 ms. M10 measured exactly that: a
  // TTFT floor of ~43 ms that did not move between a 6-token prompt and a
  // 455-token one, because it was never prefill -- it was the socket. The
  // engine's job is to put a token on the wire as soon as it exists, so the
  // batching heuristic that helps bulk transfers is precisely wrong here.
  impl->server.set_tcp_nodelay(true);

  // Generation can take much longer than a default socket timeout, and a
  // non-streaming request holds the connection open for all of it.
  impl->server.set_write_timeout(config.write_timeout_seconds, 0);

  impl->Route();

  return std::unique_ptr<HttpServer>(new HttpServer(std::move(impl)));
}

Status HttpServer::Listen() {
  // Port 0 means "any free port", which is how a test avoids colliding with
  // whatever else is on the machine. The chosen port is published before the
  // accept loop starts, so a caller that waits for readiness can read it.
  if (impl_->config.port == 0) {
    const int bound = impl_->server.bind_to_any_port(impl_->config.host);

    if (bound < 0) {
      return InternalError("cannot bind to ", impl_->config.host,
                           " on any port");
    }

    impl_->bound_port.store(bound, std::memory_order_release);
  } else {
    if (!impl_->server.bind_to_port(impl_->config.host, impl_->config.port)) {
      return InternalError("cannot bind to ", impl_->config.host, ":",
                           impl_->config.port);
    }

    impl_->bound_port.store(impl_->config.port, std::memory_order_release);
  }

  if (!impl_->server.listen_after_bind()) {
    return InternalError("listen failed");
  }

  return OkStatus();
}

void HttpServer::Stop() {
  if (impl_ != nullptr) impl_->server.stop();
}

bool HttpServer::WaitUntilReady() {
  impl_->server.wait_until_ready();

  return impl_->server.is_running();
}

int HttpServer::port() const {
  return impl_->bound_port.load(std::memory_order_acquire);
}

}  // namespace inferx::server
