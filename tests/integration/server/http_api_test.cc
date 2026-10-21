// End-to-end HTTP API tests for the vLLM-aligned server surface
// (docs/inference_server_design.md §13.2-13.4): a real HttpServer with the
// real OpenAI handlers and prompt pool on a loopback port, driven through
// a raw HTTP/1.1 client, with engine behavior scripted through
// FakeEngineClient. No engine process or GPU is involved.
#include <arpa/inet.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <cstring>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <boost/asio.hpp>
#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include "inferx/server/engine_client.h"
#include "inferx/server/http_server.h"
#include "inferx/server/metrics.h"
#include "inferx/server/openai_handlers.h"
#include "inferx/server/prompt_processor_pool.h"
#include "inferx/tokenizer/tokenizer.h"

namespace net = boost::asio;
using inferx::Status;
using inferx::StatusOr;
using inferx::EngineCompletion;
using inferx::EngineEvent;
using inferx::EngineSubmission;
using inferx::HandlerConfig;
using inferx::HttpServer;
using inferx::IEngineClient;
using inferx::IEngineEventStream;
using inferx::MetricsService;
using inferx::OpenAiHandler;
using inferx::PromptPoolConfig;
using inferx::PromptProcessorPool;
using inferx::TextPrompt;

namespace {

constexpr const char* kTokenizerPath = "tests/testdata/tiny_tokenizer/tokenizer.json";
constexpr const char* kModelName = "test-model";

EngineEvent Delta(std::string text) {
  EngineEvent e;
  e.kind = EngineEvent::Kind::kDelta;
  e.text = std::move(text);
  return e;
}

EngineEvent Finished(inferx::FinishReason reason, int prompt_tokens, int completion_tokens) {
  EngineEvent e;
  e.kind = EngineEvent::Kind::kFinished;
  e.reason = reason;
  e.prompt_tokens = prompt_tokens;
  e.completion_tokens = completion_tokens;
  return e;
}

EngineEvent ErrorEvent(bool admission, std::string message) {
  EngineEvent e;
  e.kind = EngineEvent::Kind::kError;
  e.admission = admission;
  e.error = absl::InternalError(std::move(message));
  return e;
}

/// Scripts one request: events are delivered in order with the given
/// per-event delay; exhaustion after the last event parks the caller
/// (used by disconnect tests).
class FakeEventStream : public IEngineEventStream {
 public:
  FakeEventStream(std::vector<EngineEvent> script, std::chrono::milliseconds delay)
      : script_(std::move(script)), delay_(delay) {}

  net::awaitable<StatusOr<EngineEvent>> Next() override {
    if (index_ >= script_.size()) {
      // Park: a hung generation, as far as the handler knows.
      net::steady_timer timer(co_await net::this_coro::executor);
      for (;;) {
        timer.expires_after(std::chrono::seconds(60));
        co_await timer.async_wait(net::as_tuple(net::use_awaitable));
      }
    }
    if (delay_ > std::chrono::milliseconds(0)) {
      net::steady_timer timer(co_await net::this_coro::executor);
      timer.expires_after(delay_);
      co_await timer.async_wait(net::as_tuple(net::use_awaitable));
    }
    EngineEvent event = script_[index_++];
    event.seq = seq_++;
    co_return event;
  }
  void Close() override {}

 private:
  std::vector<EngineEvent> script_;
  std::chrono::milliseconds delay_;
  std::size_t index_ = 0;
  std::uint64_t seq_ = 0;
};

using SubmitScript = std::function<
    StatusOr<std::vector<EngineEvent>>(const EngineCompletion& request)>;

class FakeEngineClient : public IEngineClient {
 public:
  std::mutex mu;
  std::vector<EngineCompletion> submitted;
  std::vector<std::string> aborted;
  SubmitScript script;
  std::chrono::milliseconds per_event_delay{0};

  StatusOr<EngineSubmission> Submit(EngineCompletion request) override {
    {
      std::lock_guard<std::mutex> lock(mu);
      submitted.push_back(std::move(request));
    }
    StatusOr<std::vector<EngineEvent>> events = script
        ? script(submitted.back())
        : StatusOr<std::vector<EngineEvent>>(std::vector<EngineEvent>{Finished(
              inferx::FinishReason::kStopped, 1, 1)});
    if (!events.ok()) return events.status();
    EngineSubmission out;
    out.request_id = "fake-" + std::to_string(++next_id_);
    out.events = std::make_unique<FakeEventStream>(*std::move(events), per_event_delay);
    return out;
  }

  void Abort(const std::string& request_id) override {
    std::lock_guard<std::mutex> lock(mu);
    aborted.push_back(request_id);
  }

 private:
  std::uint64_t next_id_ = 0;
};

/// A real server stack: HttpServer + OpenAiHandler + prompt pool + fake
/// engine, bound on an ephemeral loopback port.
class ServerHarness {
 public:
  explicit ServerHarness(std::vector<std::string> api_keys = {}, bool qwen_chat = false)
      : pool_(MakePool(io_, qwen_chat)), work_(net::make_work_guard(io_.get_executor())) {
    HandlerConfig config;
    config.model_name = kModelName;
    config.created = 1700000000;
    config.api_keys = std::move(api_keys);
    handler_.emplace(std::move(config), *pool_, engine_, metrics_,
                     /*draining=*/[] { return false; },
                     /*engine_ready=*/[] { return true; });
    server_.emplace(io_, "127.0.0.1", 0, *handler_);
    port_ = server_->Listen();
    io_thread_ = std::thread([this] { io_.run(); });
    EXPECT_TRUE(pool_->WaitUntilReady(std::chrono::milliseconds(30000)).ok());
  }

  ~ServerHarness() {
    io_.stop();
    io_thread_.join();
  }

  FakeEngineClient& engine() { return engine_; }
  std::uint16_t port() const { return port_; }

 private:
  // The pool's worker mailboxes and sweep timer bind to the harness io_,
  // which runs on io_thread_ — a pool bound to an un-run io_context would
  // never deliver completions.
  static std::shared_ptr<PromptProcessorPool> MakePool(net::io_context& io, bool qwen_chat) {
    PromptPoolConfig config;
    config.workers = 1;
    config.tokenizer_path = qwen_chat ? "tests/testdata/qwen3_chat/tokenizer.json" : kTokenizerPath;
    config.max_prompt_bytes = 64u << 20;
    config.queue_max_bytes = 64u << 20;
    return inferx::CreatePromptProcessorPool(io, config);
  }

  net::io_context io_;
  std::shared_ptr<PromptProcessorPool> pool_;
  net::executor_work_guard<net::io_context::executor_type> work_;
  FakeEngineClient engine_;
  MetricsService metrics_;
  std::optional<OpenAiHandler> handler_;
  std::optional<HttpServer> server_;
  std::thread io_thread_;
  std::uint16_t port_ = 0;
};

/// Minimal synchronous HTTP/1.1 client with chunked-body decoding.
class TestClient {
 public:
  explicit TestClient(std::uint16_t port) {
    net::ip::tcp::endpoint endpoint(net::ip::make_address("127.0.0.1"), port);
    socket_.connect(endpoint);
    timeval timeout{5, 0};
    setsockopt(socket_.native_handle(), SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
    setsockopt(socket_.native_handle(), SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout));
  }

  struct Raw {
    int status = 0;
    std::string head;
    std::string body;  // Dechunked when the response was chunked.
  };

  Raw RoundTrip(const std::string& method, const std::string& target,
                const std::string& body,
                const std::vector<std::pair<std::string, std::string>>& headers = {},
                const std::string& terminator = "") {
    // Keep-alive: the connection carries further requests in the same test.
    std::string request = method + " " + target + " HTTP/1.1\r\n";
    request += "Host: localhost\r\nConnection: keep-alive\r\n";
    request += "Content-Length: " + std::to_string(body.size()) + "\r\n";
    for (const auto& [name, value] : headers) {
      request += name + ": " + value + "\r\n";
    }
    request += "\r\n" + body;
    net::write(socket_, net::buffer(request));

    std::string raw;
    char buffer[8192];
    for (;;) {
      if (!terminator.empty() && raw.find(terminator) != std::string::npos) break;
      const std::size_t split = raw.find("\r\n\r\n");
      if (split != std::string::npos && terminator.empty()) {
        // Unary response: complete once Content-Length bytes arrived.
        const std::string head = raw.substr(0, split);
        const std::size_t cl = head.find("Content-Length:");
        if (cl != std::string::npos) {
          const unsigned long want = std::strtoul(head.c_str() + cl + 15, nullptr, 10);
          if (raw.size() - split - 4 >= want) break;
        }
      }
      boost::system::error_code ec;
      const std::size_t n = socket_.read_some(net::buffer(buffer), ec);
      if (ec || n == 0) break;
      raw.append(buffer, n);
    }

    Raw out;
    const std::size_t split = raw.find("\r\n\r\n");
    if (split == std::string::npos) return out;
    out.head = raw.substr(0, split);
    std::string body_raw = raw.substr(split + 4);
    const std::size_t space = out.head.find(' ');
    if (space != std::string::npos) {
      out.status = std::atoi(out.head.c_str() + space + 1);
    }
    out.body = out.head.find("Transfer-Encoding: chunked") != std::string::npos
                   ? Dechunk(body_raw)
                   : body_raw;
    return out;
  }

  void Close() {
    boost::system::error_code ignored;
    socket_.close(ignored);
  }

  /// Closes with SO_LINGER 0 so the peer sees a reset, not a FIN.
  void ResetHard() {
    linger option{1, 0};
    setsockopt(socket_.native_handle(), SOL_SOCKET, SO_LINGER, &option, sizeof(option));
    Close();
  }

 private:
  static std::string Dechunk(const std::string& raw) {
    std::string out;
    std::size_t pos = 0;
    while (pos < raw.size()) {
      const std::size_t eol = raw.find("\r\n", pos);
      if (eol == std::string::npos) break;
      const unsigned long size = std::strtoul(raw.substr(pos, eol - pos).c_str(), nullptr, 16);
      if (size == 0) break;
      if (eol + 2 + size > raw.size()) break;
      out.append(raw, eol + 2, size);
      pos = eol + 2 + size + 2;
    }
    return out;
  }

  net::io_context io_;
  net::ip::tcp::socket socket_{io_};
};

std::string CompletionsBody(const std::string& extra) {
  return R"({"model":")" + std::string(kModelName) + R"(","prompt":"hello world")" +
         extra + "}";
}

std::size_t CountSubstrings(const std::string& haystack, const std::string& needle) {
  std::size_t count = 0;
  for (std::size_t pos = haystack.find(needle); pos != std::string::npos;
       pos = haystack.find(needle, pos + needle.size())) {
    ++count;
  }
  return count;
}

}  // namespace

TEST(HttpApiTest, HealthLivezReadyzVersion) {
  ServerHarness harness;
  TestClient client(harness.port());

  const auto health = client.RoundTrip("GET", "/health", "");
  EXPECT_EQ(health.status, 200);
  EXPECT_TRUE(health.body.empty()) << "vLLM /health is empty-bodied";

  const auto live = client.RoundTrip("GET", "/livez", "");
  EXPECT_EQ(live.status, 200);
  EXPECT_NE(live.body.find("\"status\":\"ok\""), std::string::npos);

  const auto ready = client.RoundTrip("GET", "/readyz", "");
  EXPECT_EQ(ready.status, 200);
  EXPECT_NE(ready.body.find("\"status\":\"ok\""), std::string::npos);

  const auto version = client.RoundTrip("GET", "/version", "");
  EXPECT_EQ(version.status, 200);
  EXPECT_NE(version.body.find("version"), std::string::npos);
}

TEST(HttpApiTest, ModelsList) {
  ServerHarness harness;
  TestClient client(harness.port());
  const auto res = client.RoundTrip("GET", "/v1/models", "");
  EXPECT_EQ(res.status, 200);
  EXPECT_NE(res.body.find("\"object\":\"list\""), std::string::npos);
  EXPECT_NE(res.body.find(kModelName), std::string::npos);
}

TEST(HttpApiTest, UnknownRouteIsEnvelope404) {
  ServerHarness harness;
  TestClient client(harness.port());
  const auto res = client.RoundTrip("GET", "/v1/nope", "");
  EXPECT_EQ(res.status, 404);
  EXPECT_NE(res.body.find("\"error\""), std::string::npos);
  EXPECT_NE(res.body.find("unknown endpoint"), std::string::npos);
}

TEST(HttpApiTest, MethodNotAllowedCarriesAllow) {
  ServerHarness harness;
  TestClient client(harness.port());
  const auto res = client.RoundTrip("GET", "/v1/completions", "");
  EXPECT_EQ(res.status, 405);
  EXPECT_NE(res.head.find("Allow: POST"), std::string::npos);
  EXPECT_NE(res.body.find("\"error\""), std::string::npos);
}

TEST(HttpApiTest, AuthGuardAndPublicHealth) {
  ServerHarness harness({"secret-key"});
  TestClient client(harness.port());

  const auto no_key = client.RoundTrip("GET", "/v1/models", "");
  EXPECT_EQ(no_key.status, 401);
  EXPECT_NE(no_key.head.find("WWW-Authenticate: Bearer"), std::string::npos);

  const auto bad_key = client.RoundTrip("GET", "/v1/models", "",
                                        {{"Authorization", "Bearer wrong"}});
  EXPECT_EQ(bad_key.status, 401);

  const auto good_key = client.RoundTrip("GET", "/v1/models", "",
                                         {{"Authorization", "Bearer secret-key"}});
  EXPECT_EQ(good_key.status, 200);

  const auto health = client.RoundTrip("GET", "/health", "");
  EXPECT_EQ(health.status, 200);  // Health probes stay public.
}

TEST(HttpApiTest, NonStreamingCompletion) {
  ServerHarness harness;
  harness.engine().script = [](const EngineCompletion&) {
    return StatusOr<std::vector<EngineEvent>>(std::vector<EngineEvent>{
        Delta("Hello"), Delta(" world"), Finished(inferx::FinishReason::kStopped, 5, 2)});
  };
  TestClient client(harness.port());
  const auto res = client.RoundTrip("POST", "/v1/completions", CompletionsBody(""));
  EXPECT_EQ(res.status, 200);
  EXPECT_NE(res.body.find("\"object\":\"text_completion\""), std::string::npos);
  EXPECT_NE(res.body.find("\"text\":\"Hello world\""), std::string::npos);
  EXPECT_NE(res.body.find("\"finish_reason\":\"stop\""), std::string::npos);
  EXPECT_NE(res.body.find("\"prompt_tokens\":5"), std::string::npos);
  EXPECT_NE(res.body.find("\"completion_tokens\":2"), std::string::npos);
  ASSERT_EQ(harness.engine().submitted.size(), 1u);
  EXPECT_FALSE(harness.engine().submitted[0].prompt.empty());
  EXPECT_TRUE(harness.engine().aborted.empty());
}

TEST(HttpApiTest, StreamingFramesAndConditionalUsage) {
  ServerHarness harness;
  harness.engine().script = [](const EngineCompletion&) {
    return StatusOr<std::vector<EngineEvent>>(std::vector<EngineEvent>{
        Delta("Hel"), Delta("lo"), Finished(inferx::FinishReason::kStopped, 3, 2)});
  };
  TestClient client(harness.port());

  // Default: no usage chunk (vLLM emits it only on include_usage).
  const auto plain = client.RoundTrip("POST", "/v1/completions",
                                      CompletionsBody(",\"stream\":true"), {},
                                      "data: [DONE]");
  EXPECT_EQ(plain.status, 200);
  EXPECT_NE(plain.head.find("text/event-stream; charset=utf-8"), std::string::npos);
  EXPECT_NE(plain.head.find("X-Accel-Buffering: no"), std::string::npos);
  EXPECT_EQ(CountSubstrings(plain.body, "\"object\":\"text_completion_chunk\""), 2);
  EXPECT_NE(plain.body.find("\"text\":\"Hel\""), std::string::npos);
  EXPECT_NE(plain.body.find("\"finish_reason\":\"stop\""), std::string::npos);
  EXPECT_EQ(CountSubstrings(plain.body, "usage"), 0u);
  EXPECT_EQ(CountSubstrings(plain.body, "[DONE]"), 1u);

  TestClient usage_client(harness.port());
  harness.engine().script = [](const EngineCompletion&) {
    return StatusOr<std::vector<EngineEvent>>(std::vector<EngineEvent>{
        Delta("x"), Finished(inferx::FinishReason::kLengthCapped, 7, 4)});
  };
  const auto with_usage = usage_client.RoundTrip(
      "POST", "/v1/completions",
      CompletionsBody(",\"stream\":true,\"stream_options\":{\"include_usage\":true}"), {},
      "data: [DONE]");
  EXPECT_EQ(with_usage.status, 200);
  EXPECT_NE(with_usage.body.find("\"choices\":[],\"usage\""), std::string::npos);
  EXPECT_NE(with_usage.body.find("\"completion_tokens\":4"), std::string::npos);
  EXPECT_EQ(CountSubstrings(with_usage.body, "[DONE]"), 1u);
}

TEST(HttpApiTest, ErrorAfterHeadersIsDataFrameThenDone) {
  ServerHarness harness;
  harness.engine().script = [](const EngineCompletion&) {
    return StatusOr<std::vector<EngineEvent>>(std::vector<EngineEvent>{
        Delta("partial"), ErrorEvent(false, "boom")});
  };
  TestClient client(harness.port());
  const auto res = client.RoundTrip("POST", "/v1/completions",
                                    CompletionsBody(",\"stream\":true"), {},
                                    "data: [DONE]");
  EXPECT_EQ(res.status, 200);  // Headers committed as 200 before the error.
  EXPECT_NE(res.body.find("partial"), std::string::npos);
  EXPECT_NE(res.body.find("\"error\""), std::string::npos);
  EXPECT_EQ(CountSubstrings(res.body, "[DONE]"), 1u);
}

TEST(HttpApiTest, EngineAdmissionFailureIsRetryable503) {
  ServerHarness harness;
  harness.engine().script = [](const EngineCompletion&) {
    return StatusOr<std::vector<EngineEvent>>(
        absl::ResourceExhaustedError("engine queue full"));
  };
  TestClient client(harness.port());
  const auto res = client.RoundTrip("POST", "/v1/completions", CompletionsBody(""));
  EXPECT_EQ(res.status, 503);
  EXPECT_NE(res.head.find("Retry-After"), std::string::npos);
  EXPECT_NE(res.body.find("server_overloaded"), std::string::npos);
}

TEST(HttpApiTest, ValidationErrors) {
  ServerHarness harness;
  TestClient client(harness.port());

  const auto bad_json = client.RoundTrip("POST", "/v1/completions", "{");
  EXPECT_EQ(bad_json.status, 400);
  EXPECT_NE(bad_json.body.find("invalid_json"), std::string::npos);

  const auto bad_model =
      client.RoundTrip("POST", "/v1/completions",
                       R"({"model":"nope","prompt":"x"})");
  EXPECT_EQ(bad_model.status, 404);
  EXPECT_NE(bad_model.body.find("model_not_found"), std::string::npos);

  const auto unsupported =
      client.RoundTrip("POST", "/v1/completions",
                       R"({"model":")" + std::string(kModelName) +
                           R"(","prompt":"x","echo":true})");
  EXPECT_EQ(unsupported.status, 400);
  EXPECT_NE(unsupported.body.find("unsupported_parameter"), std::string::npos);

  const auto wrong_type = client.RoundTrip(
      "POST", "/v1/completions", CompletionsBody(""),
      {{"Content-Type", "text/plain"}});
  EXPECT_EQ(wrong_type.status, 415);
  EXPECT_NE(wrong_type.body.find("unsupported_media_type"), std::string::npos);
}

TEST(HttpApiTest, UnknownFieldsAcceptedForCompatibility) {
  ServerHarness harness;
  TestClient client(harness.port());
  const auto res = client.RoundTrip(
      "POST", "/v1/completions",
      CompletionsBody(",\"future_field\":{\"nested\":[1,2,3]}"));
  EXPECT_EQ(res.status, 200);  // vLLM/OpenAI tolerate unknown fields.
}

TEST(HttpApiTest, StopControlsReachEngine) {
  ServerHarness harness;
  bool submitted = false;
  harness.engine().script = [&](const EngineCompletion& request) {
    submitted = true;
    EXPECT_EQ(request.params.stop_strings, (std::vector<std::string>{"END"}));
    EXPECT_EQ(request.params.stop_token_ids, (std::vector<std::int32_t>{42}));
    return StatusOr<std::vector<EngineEvent>>(absl::UnavailableError("test complete"));
  };
  TestClient client(harness.port());
  const auto res = client.RoundTrip(
      "POST", "/v1/completions",
      CompletionsBody(",\"stop\":\"END\",\"stop_token_ids\":[42]"));
  EXPECT_TRUE(submitted);
  EXPECT_EQ(res.status, 503);
}

TEST(HttpApiTest, UnsupportedControlsFailBeforeGenerationIncludingStreaming) {
  ServerHarness harness;
  harness.engine().script = [](const EngineCompletion&) {
    ADD_FAILURE() << "Unsupported requests must not reach the engine";
    return StatusOr<std::vector<EngineEvent>>(std::vector<EngineEvent>{});
  };
  TestClient client(harness.port());
  for (bool stream : {false, true}) {
    for (const std::string field : {"best_of",
                                     "response_format", "tools"}) {
      SCOPED_TRACE(field);
      const auto res = client.RoundTrip(
          "POST", "/v1/completions",
          CompletionsBody(",\"" + field + "\":null,\"stream\":" +
                          (stream ? "true" : "false")));
      EXPECT_EQ(res.status, 400);
      const auto body = nlohmann::json::parse(res.body);
      EXPECT_EQ(body["error"]["code"], "unsupported_parameter");
      EXPECT_EQ(body["error"]["param"], field);
      EXPECT_EQ(body["error"]["type"], "invalid_request_error");
      EXPECT_EQ(res.body.find("data: "), std::string::npos);
    }
  }
}

TEST(HttpApiTest, TokenIdPromptPassesThrough) {
  ServerHarness harness;
  // Token ids must exist in the tiny fixture's vocabulary; take them from
  // the facade so the test encodes real ids.
  auto facade = inferx::Tokenizer::FromFile(kTokenizerPath);
  ASSERT_TRUE(facade.ok()) << facade.status();
  const auto encoded = (*facade)->Encode("hello");
  ASSERT_TRUE(encoded.ok()) << encoded.status();
  ASSERT_FALSE(encoded->empty());

  std::string body = R"({"model":")" + std::string(kModelName) + R"(","prompt":[)";
  for (std::size_t i = 0; i < encoded->size(); ++i) {
    body += std::to_string((*encoded)[i]);
    if (i + 1 < encoded->size()) body += ",";
  }
  body += "]}";

  TestClient client(harness.port());
  const auto res = client.RoundTrip("POST", "/v1/completions", body);
  EXPECT_EQ(res.status, 200) << res.body;
  ASSERT_EQ(harness.engine().submitted.size(), 1u);
  EXPECT_EQ(harness.engine().submitted[0].prompt, *encoded);
}

TEST(HttpApiTest, ChatCompletionsIsExplicitlyNotImplemented) {
  ServerHarness harness;
  TestClient client(harness.port());
  const auto res = client.RoundTrip(
      "POST", "/v1/chat/completions",
      R"({"model":")" + std::string(kModelName) + R"(","messages":[{"role":"user","content":"hi"}]})");
  EXPECT_EQ(res.status, 501);
  EXPECT_NE(res.body.find("chat_template_not_supported"), std::string::npos);

  const auto bad_model = client.RoundTrip(
      "POST", "/v1/chat/completions",
      R"({"model":"nope","messages":[{"role":"user","content":"hi"}]})");
  EXPECT_EQ(bad_model.status, 404);
}

TEST(HttpApiTest, ChatUnaryAndStreamingUseChatSchemaAndTemplatedPrompt) {
  ServerHarness harness({}, true);
  harness.engine().script = [](const EngineCompletion&) {
    return StatusOr<std::vector<EngineEvent>>(std::vector<EngineEvent>{
        Delta("hello"), Delta(" world"), Finished(inferx::FinishReason::kStopped, 8, 2)});
  };
  TestClient client(harness.port());
  const std::string base = R"({"messages":[{"role":"user","content":"hi"}],"chat_template_kwargs":{"enable_thinking":false})";
  auto plain = client.RoundTrip("POST", "/v1/chat/completions", base + "}");
  ASSERT_EQ(plain.status, 200) << plain.body;
  auto json = nlohmann::json::parse(plain.body);
  EXPECT_EQ(json["object"], "chat.completion");
  EXPECT_EQ(json["choices"][0]["message"]["role"], "assistant");
  EXPECT_EQ(json["choices"][0]["message"]["content"], "hello world");
  EXPECT_EQ(json["usage"]["completion_tokens"], 2);
  auto tokenizer = inferx::Tokenizer::FromFile("tests/testdata/qwen3_chat/tokenizer.json");
  ASSERT_TRUE(tokenizer.ok());
  auto expected = (*tokenizer)->Encode("<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n");
  ASSERT_TRUE(expected.ok());
  ASSERT_EQ(harness.engine().submitted.size(), 1u);
  EXPECT_EQ(harness.engine().submitted[0].prompt, *expected);
  auto stream = client.RoundTrip("POST", "/v1/chat/completions",
      base + R"(,"stream":true,"stream_options":{"include_usage":true}})", {}, "data: [DONE]");
  ASSERT_EQ(stream.status, 200) << stream.body;
  EXPECT_NE(stream.body.find("chat.completion.chunk"), std::string::npos);
  EXPECT_NE(stream.body.find("\"role\":\"assistant\""), std::string::npos);
  EXPECT_NE(stream.body.find("\"content\":\"hello\""), std::string::npos);
  EXPECT_NE(stream.body.find("\"content\":\" world\""), std::string::npos);
  EXPECT_NE(stream.body.find("\"choices\":[]"), std::string::npos);
  EXPECT_NE(stream.body.find("data: [DONE]"), std::string::npos);
  EXPECT_EQ(stream.body.find("text_completion"), std::string::npos);
}

TEST(HttpApiTest, DisconnectAbortsEngineRequest) {
  ServerHarness harness;
  harness.engine().per_event_delay = std::chrono::milliseconds(50);
  harness.engine().script = [](const EngineCompletion&) {
    return StatusOr<std::vector<EngineEvent>>(std::vector<EngineEvent>{
        Delta("a"), Delta("b"), Delta("c"), Delta("d"), Delta("e"),
        Finished(inferx::FinishReason::kStopped, 1, 5)});
  };
  TestClient client(harness.port());
  // Read only until the first delta arrives, then RST the connection
  // (SO_LINGER 0) so the server's next SSE write fails rather than
  // draining into the OS buffer.
  const auto res = client.RoundTrip("POST", "/v1/completions",
                                    CompletionsBody(",\"stream\":true"), {},
                                    "\"text\":\"a\"");
  EXPECT_EQ(res.status, 200);
  client.ResetHard();

  // The failed write aborts the engine request exactly once (docs §6.6).
  bool aborted = false;
  for (int i = 0; i < 200; ++i) {
    {
      std::lock_guard<std::mutex> lock(harness.engine().mu);
      aborted = !harness.engine().aborted.empty();
    }
    if (aborted) break;
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  EXPECT_TRUE(aborted);
  std::lock_guard<std::mutex> lock(harness.engine().mu);
  EXPECT_EQ(harness.engine().aborted.size(), 1u);
}

TEST(HttpApiTest, MetricsEndpointExposesCounters) {
  ServerHarness harness;
  TestClient warm(harness.port());
  EXPECT_EQ(warm.RoundTrip("GET", "/v1/models", "").status, 200);

  TestClient client(harness.port());
  const auto res = client.RoundTrip("GET", "/metrics", "");
  EXPECT_EQ(res.status, 200);
  EXPECT_NE(res.head.find("text/plain"), std::string::npos);
  EXPECT_NE(res.body.find("inferx_http_requests_total"), std::string::npos);
  EXPECT_NE(res.body.find("/v1/models"), std::string::npos);
}
