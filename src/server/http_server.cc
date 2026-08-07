#include "inferx/server/http_server.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <future>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#include <boost/asio/ip/tcp.hpp>
#include <boost/asio/post.hpp>
#include <boost/asio/strand.hpp>
#include <boost/asio/steady_timer.hpp>
#include <boost/asio/thread_pool.hpp>
#include <boost/asio/write.hpp>
#include <boost/beast/core.hpp>
#include <boost/beast/http.hpp>
#include <openssl/crypto.h>
#include <openssl/sha.h>

#include "inferx/api/openai.h"
#include "inferx/observe/metrics.h"
#include "inferx/support/json.h"
#include "inferx/tokenizer/chat_template.h"

namespace inferx::server {
namespace {

namespace asio = boost::asio;
namespace beast = boost::beast;
namespace http = beast::http;
using tcp = asio::ip::tcp;
using api::FinishReason;
using api::SamplingRequest;
using api::Usage;

int64_t NowSeconds() {
  return std::chrono::duration_cast<std::chrono::seconds>(
             std::chrono::system_clock::now().time_since_epoch())
      .count();
}

std::string MakeId(std::string_view prefix) {
  static std::atomic<uint64_t> counter{0};
  return std::string(prefix) + "-" +
         std::to_string(counter.fetch_add(1, std::memory_order_relaxed));
}

// Sortable by ingress time and collision-resistant across processes. The
// random process nonce keeps independently started gateways in separate ID
// spaces; the counter orders requests received within the same millisecond.
std::string MakeRequestId() {
  static const uint32_t process_nonce = [] {
    const auto now = std::chrono::high_resolution_clock::now()
                         .time_since_epoch()
                         .count();
    const auto address = reinterpret_cast<uintptr_t>(&MakeRequestId);
    return static_cast<uint32_t>(now ^ (now >> 32) ^ address);
  }();
  static std::atomic<uint32_t> counter{0};
  const uint64_t milliseconds =
      std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::system_clock::now().time_since_epoch())
          .count();
  char id[5 + 12 + 8 + 8 + 1];
  std::snprintf(id, sizeof(id), "req_%012llx%08x%08x",
                static_cast<unsigned long long>(milliseconds), process_nonce,
                counter.fetch_add(1, std::memory_order_relaxed));
  return id;
}

std::string ErrorJson(std::string_view message, std::string_view type,
                      std::string_view code, std::string_view request_id) {
  std::string out = "{\"error\":{\"message\":";
  AppendJsonString(message, &out);
  out += ",\"type\":";
  AppendJsonString(type, &out);
  out += ",\"param\":null,\"code\":";
  if (code.empty()) {
    out += "null";
  } else {
    AppendJsonString(code, &out);
  }
  out += ",\"request_id\":";
  AppendJsonString(request_id, &out);
  out += "}}";
  return out;
}

FinishReason ToApiReason(scheduler::FinishReason reason) {
  return reason == scheduler::FinishReason::kStopToken ? FinishReason::kStop
                                                       : FinishReason::kLength;
}

scheduler::SamplingParams ToSchedulerParams(const SamplingRequest& request) {
  static std::atomic<uint64_t> counter{0x243f6a8885a308d3ULL};
  scheduler::SamplingParams params;
  params.temperature = request.temperature;
  params.top_p = request.top_p;
  params.seed = request.has_seed
                    ? request.seed
                    : counter.fetch_add(0x9e3779b97f4a7c15ULL,
                                        std::memory_order_relaxed);
  return params;
}

int StatusToHttp(const Status& status) {
  switch (status.code()) {
    case absl::StatusCode::kUnauthenticated:
      return 401;
    case absl::StatusCode::kPermissionDenied:
      return 403;
    case absl::StatusCode::kInvalidArgument:
      return 400;
    case absl::StatusCode::kUnimplemented:
      return 422;
    case absl::StatusCode::kResourceExhausted:
      return 429;
    case absl::StatusCode::kNotFound:
      return 404;
    case absl::StatusCode::kAlreadyExists:
    case absl::StatusCode::kFailedPrecondition:
    case absl::StatusCode::kAborted:
      return 409;
    case absl::StatusCode::kUnavailable:
      return 503;
    case absl::StatusCode::kDeadlineExceeded:
      return 504;
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

size_t DefaultThreads(size_t requested, size_t maximum) {
  if (requested != 0) return requested;
  return std::max<size_t>(1, std::min<size_t>(maximum,
      std::max(1u, std::thread::hardware_concurrency())));
}

bool DecodeSha256(
    std::string_view hex,
    std::array<unsigned char, SHA256_DIGEST_LENGTH>* out) {
  if (hex.size() != SHA256_DIGEST_LENGTH * 2) return false;
  auto digit = [](char value) -> int {
    if (value >= '0' && value <= '9') return value - '0';
    if (value >= 'a' && value <= 'f') return value - 'a' + 10;
    if (value >= 'A' && value <= 'F') return value - 'A' + 10;
    return -1;
  };
  for (size_t index = 0; index < out->size(); ++index) {
    const int high = digit(hex[index * 2]);
    const int low = digit(hex[index * 2 + 1]);
    if (high < 0 || low < 0) return false;
    (*out)[index] = static_cast<unsigned char>((high << 4) | low);
  }
  return true;
}

}  // namespace

struct HttpServer::Impl {
  struct Session;

  Engine* engine;
  HttpServerConfig config;
  asio::io_context io;
  tcp::acceptor acceptor{io};
  asio::thread_pool application_pool;
  std::vector<std::thread> io_workers;
  std::atomic<int> bound_port{0};
  std::atomic<bool> stopping{false};
  std::atomic<size_t> active_requests{0};
  std::vector<std::array<unsigned char, SHA256_DIGEST_LENGTH>> api_key_hashes;
  std::mutex ready_mutex;
  std::condition_variable ready_cv;
  bool ready = false;
  std::mutex sessions_mutex;
  std::vector<std::weak_ptr<Session>> sessions;

  Impl(Engine* value, HttpServerConfig settings)
      : engine(value),
        config(std::move(settings)),
        application_pool(DefaultThreads(config.application_threads, 32)) {
    api_key_hashes.reserve(config.api_key_sha256.size());
    for (const std::string& encoded : config.api_key_sha256) {
      std::array<unsigned char, SHA256_DIGEST_LENGTH> decoded{};
      DecodeSha256(encoded, &decoded);
      api_key_hashes.push_back(decoded);
    }
  }

  ~Impl() {
    Stop();
    application_pool.join();
  }

  void Accept();
  Status Listen();

  void Register(const std::shared_ptr<Session>& session) {
    std::lock_guard lock(sessions_mutex);
    sessions.emplace_back(session);
  }

  void Stop();
};

struct HttpServer::Impl::Session
    : std::enable_shared_from_this<HttpServer::Impl::Session> {
  Impl* owner;
  beast::tcp_stream stream;
  beast::flat_buffer buffer;
  std::unique_ptr<http::request_parser<http::string_body>> parser;
  std::shared_ptr<Generation> generation;
  std::mutex generation_mutex;
  asio::steady_timer request_timer;
  bool keep_alive = false;
  bool closed = false;
  std::atomic<bool> admitted{false};
  std::string request_id;
  std::string client_request_id;

  Session(Impl* value, tcp::socket socket)
      : owner(value),
        stream(std::move(socket)),
        request_timer(stream.get_executor()) {
  }

  void Run() {
    beast::error_code error;
    stream.socket().set_option(tcp::no_delay(true), error);
    Read();
  }

  void Read() {
    if (owner->stopping || closed) return Close();
    parser = std::make_unique<http::request_parser<http::string_body>>();
    parser->body_limit(owner->config.max_request_bytes);
    stream.expires_after(std::chrono::seconds(owner->config.read_timeout_seconds));
    http::async_read(stream, buffer, *parser,
                     beast::bind_front_handler(&Session::OnRead,
                                               shared_from_this()));
  }

  void OnRead(beast::error_code error, size_t) {
    if (error == http::error::end_of_stream) return Close();
    if (error == http::error::body_limit) {
      return WriteError(413, "request body too large", "invalid_request_error",
                        false);
    }
    if (error) return Close();
    http::request<http::string_body> request = parser->release();
    keep_alive = request.keep_alive();
    request_id = MakeRequestId();
    if (const auto supplied = request.find("X-Request-ID");
        supplied != request.end()) {
      client_request_id = supplied->value();
    } else {
      client_request_id.clear();
    }
    const size_t previous =
        owner->active_requests.fetch_add(1, std::memory_order_acq_rel);
    if (previous >= owner->config.max_active_requests) {
      owner->active_requests.fetch_sub(1, std::memory_order_acq_rel);
      return WriteError(429, "HTTP request capacity exceeded",
                        "rate_limit_error", keep_alive,
                        "request_capacity_exceeded");
    }
    admitted.store(true, std::memory_order_release);
    auto self = shared_from_this();
    asio::post(owner->application_pool,
               [self, request = std::move(request)]() mutable {
                 self->Handle(std::move(request));
               });
  }

  void Handle(http::request<http::string_body> request) {
    try {
      const std::string target(request.target());
      if (request.method() == http::verb::get && target == "/health") {
        return WriteJson(200, "{\"status\":\"ok\"}");
      }
      if (request.method() == http::verb::get && target == "/health/live") {
        return WriteJson(200, "{\"status\":\"ok\"}");
      }
      if (request.method() == http::verb::get && target == "/health/ready") {
        return WriteJson(owner->stopping ? 503 : 200,
                         owner->stopping ? "{\"status\":\"not_ready\"}"
                                         : "{\"status\":\"ready\"}");
      }
      if (request.method() == http::verb::get && target == "/health/startup") {
        return WriteJson(200, "{\"status\":\"started\"}");
      }
      if (!Authenticate(request)) {
        return WriteError(401, "missing or invalid bearer token",
                          "authentication_error", keep_alive,
                          "invalid_api_key");
      }
      if (request.method() == http::verb::get && target == "/v1/models") {
        return WriteJson(200, api::ModelsJson(owner->engine->model_name(),
                                               NowSeconds()));
      }
      if (request.method() == http::verb::get && target == "/stats") {
        return WriteJson(200, RenderStatsJson(owner->engine->stats()));
      }
      if (request.method() == http::verb::get && target == "/metrics") {
        return WriteText(200, owner->engine->metrics() +
                                  RenderMetrics(owner->engine->stats()),
                         "text/plain; version=0.0.4; charset=utf-8");
      }
      if (request.method() == http::verb::post &&
          (target == "/v1/chat/completions" ||
           target == "/v1/completions")) {
        return HandleCompletion(request.body(),
                                target == "/v1/chat/completions");
      }
      if (target == "/v1/chat/completions" ||
          target == "/v1/completions") {
        keep_alive = request.keep_alive();
        return WriteError(405, "method not allowed", "invalid_request_error",
                          keep_alive, "method_not_allowed");
      }
      WriteError(404, "route not found", "invalid_request_error", keep_alive);
    } catch (...) {
      WriteError(500, "internal error", "server_error", keep_alive);
    }
  }

  bool Authenticate(const http::request<http::string_body>& request) const {
    if (owner->api_key_hashes.empty()) return true;
    const auto authorization = request.find(http::field::authorization);
    if (authorization == request.end()) return false;
    const beast::string_view value = authorization->value();
    constexpr std::string_view prefix = "Bearer ";
    if (value.size() <= prefix.size() ||
        std::string_view(value.data(), prefix.size()) != prefix) {
      return false;
    }
    std::array<unsigned char, SHA256_DIGEST_LENGTH> digest{};
    SHA256(reinterpret_cast<const unsigned char*>(value.data() + prefix.size()),
           value.size() - prefix.size(), digest.data());
    unsigned int matched = 0;
    for (const auto& accepted : owner->api_key_hashes) {
      matched |= static_cast<unsigned int>(
          CRYPTO_memcmp(digest.data(), accepted.data(), digest.size()) == 0);
    }
    return matched != 0;
  }

  void HandleCompletion(const std::string& body, bool chat) {
    SamplingRequest sampling;
    std::vector<int32_t> ids;
    if (chat) {
      auto parsed = api::ParseChatCompletionRequest(body);
      if (!parsed.ok()) return WriteStatusError(parsed.status());
      if (parsed->model != owner->engine->model_name()) {
        return WriteError(404, "model is not available: " + parsed->model,
                          "invalid_request_error", keep_alive,
                          "model_not_found");
      }
      auto prompt = tokenizer::ApplyQwen2ChatTemplate(
          parsed->messages, /*add_generation_prompt=*/true);
      if (!prompt.ok()) return WriteStatusError(prompt.status());
      auto request_tokenizer = owner->engine->tokenizer().Clone();
      if (!request_tokenizer.ok())
        return WriteError(500, request_tokenizer.status().message(),
                          "server_error", keep_alive);
      tokenizer::EncodeOptions options;
      options.special_tokens = tokenizer::SpecialTokenMode::kAsControl;
      auto encoded = (*request_tokenizer)->EncodeWithOptions(*prompt, options);
      if (!encoded.ok()) return WriteStatusError(encoded.status());
      sampling = parsed->sampling;
      ids = std::move(*encoded);
    } else {
      auto parsed = api::ParseCompletionRequest(body);
      if (!parsed.ok()) return WriteStatusError(parsed.status());
      if (parsed->model != owner->engine->model_name()) {
        return WriteError(404, "model is not available: " + parsed->model,
                          "invalid_request_error", keep_alive,
                          "model_not_found");
      }
      auto request_tokenizer = owner->engine->tokenizer().Clone();
      if (!request_tokenizer.ok())
        return WriteError(500, request_tokenizer.status().message(),
                          "server_error", keep_alive);
      auto encoded = (*request_tokenizer)->EncodeWithOptions(parsed->prompt, {});
      if (!encoded.ok()) return WriteStatusError(encoded.status());
      if (encoded->empty())
        return WriteError(400, "prompt encodes to no tokens",
                          "invalid_request_error", keep_alive);
      sampling = parsed->sampling;
      ids = std::move(*encoded);
    }

    auto submitted = owner->engine->Submit(std::move(ids), sampling.max_tokens,
                                            sampling.stop,
                                            ToSchedulerParams(sampling));
    if (!submitted.ok()) return WriteStatusError(submitted.status());
    {
      std::lock_guard lock(generation_mutex);
      generation = *submitted;
    }
    ArmRequestDeadline(*submitted);
    if (sampling.stream) {
      Stream(sampling, chat, *submitted);
    } else {
      Collect(sampling, chat, *submitted);
    }
  }

  void Collect(const SamplingRequest& sampling, bool chat,
               const std::shared_ptr<Generation>& current) {
    std::string text;
    Generation::Event event;
    scheduler::FinishReason reason = scheduler::FinishReason::kNotFinished;
    int32_t generated = 0;
    while (current->Next(&event)) {
      if (event.done) {
        reason = event.reason;
        generated = event.generated;
        break;
      }
      text += event.text;
      generated = event.generated;
    }
    Usage usage{current->prompt_tokens(), generated};
    const std::string id = MakeId(chat ? "chatcmpl" : "cmpl");
    const int64_t created = NowSeconds();
    const bool sampled = sampling.temperature > 0.0f;
    ClearGeneration();
    WriteJson(200, chat ? api::ChatCompletionJson(
                              id, owner->engine->model_name(), text,
                              ToApiReason(reason), usage, created, sampled)
                        : api::CompletionJson(id, owner->engine->model_name(),
                                              text, ToApiReason(reason), usage,
                                              created, sampled));
  }

  void Stream(const SamplingRequest& sampling, bool chat,
              const std::shared_ptr<Generation>& current) {
    const std::string id = MakeId(chat ? "chatcmpl" : "cmpl");
    const std::string model = owner->engine->model_name();
    const int64_t created = NowSeconds();
    const bool sampled = sampling.temperature > 0.0f;
    if (!StartStream()) return current->Cancel();
    if (chat && !WriteChunk(api::SseFrame(api::ChatCompletionChunkJson(
                    id, model, "assistant", "", nullptr, created, sampled)))) {
      return current->Cancel();
    }
    Generation::Event event;
    while (current->Next(&event)) {
      std::string frame;
      if (event.done) {
        const FinishReason reason = ToApiReason(event.reason);
        frame = api::SseFrame(chat ? api::ChatCompletionChunkJson(
                                        id, model, "", "", &reason, created,
                                        sampled)
                                  : api::CompletionChunkJson(
                                        id, model, "", &reason, created,
                                        sampled));
        if (sampling.include_usage) {
          Usage usage{current->prompt_tokens(), event.generated};
          frame += api::SseFrame(api::UsageChunkJson(
              id, model, usage, created, chat, sampled));
        }
        frame += api::SseFrame("[DONE]");
        if (!WriteChunk(std::move(frame))) current->Cancel();
        ClearGeneration();
        FinishStream();
        return;
      }
      if (!event.text.empty()) {
        frame = api::SseFrame(chat ? api::ChatCompletionChunkJson(
                                        id, model, "", event.text, nullptr,
                                        created, sampled)
                                  : api::CompletionChunkJson(
                                        id, model, event.text, nullptr, created,
                                        sampled));
        if (!WriteChunk(std::move(frame))) return current->Cancel();
      }
    }
    current->Cancel();
    ClearGeneration();
    FinishStream();
  }

  void ClearGeneration() {
    {
      std::lock_guard lock(generation_mutex);
      generation.reset();
    }
    auto self = shared_from_this();
    asio::post(stream.get_executor(), [self] {
      self->request_timer.cancel();
    });
  }

  void ReleaseAdmission() {
    if (admitted.exchange(false, std::memory_order_acq_rel)) {
      owner->active_requests.fetch_sub(1, std::memory_order_acq_rel);
    }
  }

  void ArmRequestDeadline(const std::shared_ptr<Generation>& current) {
    auto self = shared_from_this();
    asio::post(stream.get_executor(), [self, current] {
      self->request_timer.expires_after(std::chrono::seconds(
          self->owner->config.request_timeout_seconds));
      self->request_timer.async_wait([self, current](beast::error_code error) {
        if (!error) current->Cancel();
      });
    });
  }

  void WriteStatusError(const Status& status) {
    std::string_view code = "internal_error";
    if (status.code() == absl::StatusCode::kInvalidArgument) {
      code = "invalid_request";
    } else if (status.code() == absl::StatusCode::kResourceExhausted) {
      code = "capacity_exceeded";
    } else if (status.code() == absl::StatusCode::kUnavailable) {
      code = "model_unavailable";
    } else if (status.code() == absl::StatusCode::kDeadlineExceeded) {
      code = "inference_timeout";
    }
    WriteError(StatusToHttp(status), status.message(),
               status.code() == absl::StatusCode::kResourceExhausted
                   ? "rate_limit_error"
                   : "invalid_request_error",
               keep_alive, code);
  }

  void WriteError(int status, std::string_view message, std::string_view type,
                  bool preserve_connection, std::string_view code = {}) {
    keep_alive = preserve_connection;
    WriteJson(status, ErrorJson(message, type, code, request_id));
  }

  void WriteJson(int status, std::string body) {
    WriteText(status, std::move(body), "application/json");
  }

  void WriteText(int status, std::string body, std::string content_type) {
    auto response = std::make_shared<http::response<http::string_body>>(
        static_cast<http::status>(status), 11);
    response->set(http::field::server, "InferX");
    response->set(http::field::content_type, std::move(content_type));
    response->set("X-Request-ID", request_id);
    if (status == 401) {
      response->set(http::field::www_authenticate, "Bearer");
    }
    if (status == 429 || status == 503) response->set(http::field::retry_after, "1");
    response->keep_alive(keep_alive && !owner->stopping);
    response->body() = std::move(body);
    response->prepare_payload();
    auto self = shared_from_this();
    asio::post(stream.get_executor(), [self, response] {
      self->stream.expires_after(
          std::chrono::seconds(self->owner->config.write_timeout_seconds));
      http::async_write(self->stream, *response,
                        [self, response](beast::error_code error, size_t) {
                          self->ReleaseAdmission();
                          if (error || !response->keep_alive()) self->Close();
                          else self->Read();
                        });
    });
  }

  bool StartStream() {
    auto promise = std::make_shared<std::promise<bool>>();
    auto future = promise->get_future();
    auto head = std::make_shared<http::response<http::empty_body>>(
        http::status::ok, 11);
    head->set(http::field::server, "InferX");
    head->set(http::field::content_type, "text/event-stream");
    head->set("X-Request-ID", request_id);
    head->set(http::field::cache_control, "no-cache");
    head->set("X-Accel-Buffering", "no");
    head->keep_alive(keep_alive && !owner->stopping);
    head->chunked(true);
    auto serializer =
        std::make_shared<http::response_serializer<http::empty_body>>(*head);
    auto self = shared_from_this();
    asio::post(stream.get_executor(), [self, head, serializer, promise] {
      self->stream.expires_after(
          std::chrono::seconds(self->owner->config.write_timeout_seconds));
      http::async_write_header(
          self->stream, *serializer,
          [self, head, serializer, promise](beast::error_code error, size_t) {
            if (error) self->Close();
            promise->set_value(!error);
          });
    });
    return future.get();
  }

  bool WriteChunk(std::string data) {
    auto promise = std::make_shared<std::promise<bool>>();
    auto future = promise->get_future();
    auto payload = std::make_shared<std::string>(std::move(data));
    auto self = shared_from_this();
    asio::post(stream.get_executor(), [self, payload, promise] {
      self->stream.expires_after(
          std::chrono::seconds(self->owner->config.write_timeout_seconds));
      asio::async_write(
          self->stream, http::make_chunk(asio::buffer(*payload)),
          [self, payload, promise](beast::error_code error, size_t) {
            if (error) self->Close();
            promise->set_value(!error);
          });
    });
    return future.get();
  }

  void FinishStream() {
    auto self = shared_from_this();
    asio::post(stream.get_executor(), [self] {
      self->stream.expires_after(
          std::chrono::seconds(self->owner->config.write_timeout_seconds));
      asio::async_write(
          self->stream, http::make_chunk_last(),
          [self](beast::error_code error, size_t) {
            self->ReleaseAdmission();
            if (error || !self->keep_alive || self->owner->stopping)
              self->Close();
            else
              self->Read();
          });
    });
  }

  void Close() {
    if (closed) return;
    closed = true;
    ReleaseAdmission();
    {
      std::lock_guard lock(generation_mutex);
      if (generation) generation->Cancel();
      generation.reset();
    }
    beast::error_code ignored;
    request_timer.cancel();
    stream.socket().shutdown(tcp::socket::shutdown_both, ignored);
    stream.socket().close(ignored);
  }
};

void HttpServer::Impl::Stop() {
  if (stopping.exchange(true)) return;
  std::vector<std::shared_ptr<Session>> active;
  {
    std::lock_guard lock(sessions_mutex);
    for (auto& weak : sessions) {
      if (auto session = weak.lock()) active.emplace_back(std::move(session));
    }
  }
  asio::post(io, [this, active = std::move(active)] {
    beast::error_code ignored;
    acceptor.cancel(ignored);
    acceptor.close(ignored);
    for (const auto& session : active) session->Close();
  });
  application_pool.stop();
  ready_cv.notify_all();
}

void HttpServer::Impl::Accept() {
  acceptor.async_accept(asio::make_strand(io), [this](beast::error_code error,
                                                       tcp::socket socket) {
    if (!error) {
      auto session = std::make_shared<Session>(this, std::move(socket));
      Register(session);
      session->Run();
    }
    if (!stopping) Accept();
  });
}

Status HttpServer::Impl::Listen() {
  beast::error_code error;
  const auto address = asio::ip::make_address(config.host, error);
  if (error) return InvalidArgumentError("invalid listen address: ", config.host);
  tcp::endpoint endpoint(address, static_cast<unsigned short>(config.port));
  acceptor.open(endpoint.protocol(), error);
  if (error) return InternalError("cannot open listener: ", error.message());
  acceptor.set_option(asio::socket_base::reuse_address(true), error);
  acceptor.bind(endpoint, error);
  if (error) return InternalError("cannot bind to ", config.host, ":",
                                  config.port, ": ", error.message());
  acceptor.listen(asio::socket_base::max_listen_connections, error);
  if (error) return InternalError("listen failed: ", error.message());
  bound_port.store(acceptor.local_endpoint().port(), std::memory_order_release);
  {
    std::lock_guard lock(ready_mutex);
    ready = true;
  }
  ready_cv.notify_all();
  Accept();

  const size_t count = DefaultThreads(config.io_threads, 8);
  io_workers.reserve(count > 0 ? count - 1 : 0);
  for (size_t index = 1; index < count; ++index) {
    io_workers.emplace_back([this] { io.run(); });
  }
  io.run();
  for (auto& worker : io_workers) {
    if (worker.joinable()) worker.join();
  }
  io_workers.clear();
  return OkStatus();
}

HttpServer::HttpServer(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
HttpServer::~HttpServer() { Stop(); }

StatusOr<std::unique_ptr<HttpServer>> HttpServer::Create(
    Engine* engine, const HttpServerConfig& config) {
  if (engine == nullptr) return InvalidArgumentError("engine is null");
  if (config.port < 0 || config.port > 65535)
    return InvalidArgumentError("port must be in [0, 65535]");
  if (config.max_request_bytes == 0)
    return InvalidArgumentError("max_request_bytes must be positive");
  if (config.max_active_requests == 0)
    return InvalidArgumentError("max_active_requests must be positive");
  for (const std::string& encoded : config.api_key_sha256) {
    std::array<unsigned char, SHA256_DIGEST_LENGTH> decoded{};
    if (!DecodeSha256(encoded, &decoded)) {
      return InvalidArgumentError(
          "api_key_sha256 entries must contain exactly 64 hexadecimal "
          "characters");
    }
  }
  if (config.read_timeout_seconds <= 0 || config.write_timeout_seconds <= 0 ||
      config.request_timeout_seconds <= 0)
    return InvalidArgumentError("HTTP timeouts must be positive");
  return std::unique_ptr<HttpServer>(
      new HttpServer(std::make_unique<Impl>(engine, config)));
}

Status HttpServer::Listen() { return impl_->Listen(); }
void HttpServer::Stop() {
  if (impl_) impl_->Stop();
}
bool HttpServer::WaitUntilReady() {
  std::unique_lock lock(impl_->ready_mutex);
  impl_->ready_cv.wait(lock,
                       [this] { return impl_->ready || impl_->stopping; });
  return impl_->ready && !impl_->stopping;
}
int HttpServer::port() const {
  return impl_->bound_port.load(std::memory_order_acquire);
}

}  // namespace inferx::server
