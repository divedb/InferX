#include <folly/CancellationToken.h>
#include <folly/coro/BlockingWait.h>
#include <folly/coro/Task.h>
#include <folly/executors/CPUThreadPoolExecutor.h>
#include <folly/futures/ManualTimekeeper.h>
#include <gtest/gtest.h>

#include <boost/asio.hpp>
#include <boost/beast.hpp>
#include <memory>
#include <string>
#include <thread>
#include <utility>

#include "inferx/api/openai.h"
#include "inferx/server/admission/admission_controller.h"
#include "inferx/server/admission/capacity_policy.h"
#include "inferx/server/api/error_mapping.h"
#include "inferx/server/auth/api_key_store.h"
#include "inferx/server/auth/authenticator.h"
#include "inferx/server/auth/rbac.h"
#include "inferx/server/config/server_config.h"
#include "inferx/server/coroutine/deadline.h"
#include "inferx/server/handlers/admin_model_handler.h"
#include "inferx/server/handlers/api_routes.h"
#include "inferx/server/handlers/chat_completion_handler.h"
#include "inferx/server/handlers/completion_handler.h"
#include "inferx/server/handlers/embeddings_handler.h"
#include "inferx/server/handlers/health_handler.h"
#include "inferx/server/handlers/metrics_handler.h"
#include "inferx/server/handlers/models_handler.h"
#include "inferx/server/handlers/tokenize_handler.h"
#include "inferx/server/handlers/vllm_compat_handler.h"
#include "inferx/server/middleware/authentication.h"
#include "inferx/server/model_registry/registry.h"
#include "inferx/server/request/managed_request_service.h"
#include "inferx/server/request/request_context.h"
#include "inferx/server/request/request_id.h"
#include "inferx/server/request/request_manager.h"
#include "inferx/server/scheduler_client/in_process_scheduler_client.h"
#include "inferx/server/streaming/backpressure_controller.h"
#include "inferx/server/streaming/event_buffer.h"
#include "inferx/server/streaming/event_router.h"
#include "inferx/server/tokenization/tokenization_service.h"
#include "inferx/server/transport/beast_folly_adapter.h"
#include "inferx/server/transport/beast_listener.h"
#include "inferx/server/transport/io_runtime.h"
#include "inferx/server/transport/routes.h"
#include "inferx/server/transport/sse_writer.h"

namespace inferx::server::transport {
namespace {

namespace asio = boost::asio;
namespace beast = boost::beast;
namespace http = beast::http;
using tcp = asio::ip::tcp;

class CapturingWriter final : public ResponseWriter {
 public:
  folly::coro::Task<Status> WriteResponse(HttpResponse response,
                                          folly::CancellationToken) override {
    status = response.result_int();
    body = std::move(response.body());
    co_return OkStatus();
  }
  folly::coro::Task<Status> StartStream(HttpStreamHead response,
                                        folly::CancellationToken) override {
    status = response.result_int();
    co_return OkStatus();
  }
  folly::coro::Task<Status> Write(std::string data,
                                  folly::CancellationToken) override {
    body += data;
    co_return OkStatus();
  }
  folly::coro::Task<Status> Finish(folly::CancellationToken) override {
    finished = true;
    co_return OkStatus();
  }

  int status = 0;
  std::string body;
  bool finished = false;
};

class OkHandler final : public RequestHandler {
 public:
  folly::coro::Task<void> Handle(
      HttpRequest request, RequestContext context, ResponseWriter& writer,
      folly::CancellationToken cancellation) override {
    last_context = std::move(context);
    HttpResponse response{http::status::ok, request.version()};
    response.set(http::field::content_type, "application/json");
    response.keep_alive(request.keep_alive());
    response.body() = "{\"ok\":true}";
    response.prepare_payload();
    (void)co_await writer.WriteResponse(std::move(response), cancellation);
  }
  RequestContext last_context;
};

/// Writes the raw request target back, so a test can see which route matched
/// and what remainder a prefix route's handler would parse.
class TargetEchoHandler final : public RequestHandler {
 public:
  folly::coro::Task<void> Handle(
      HttpRequest request, RequestContext, ResponseWriter& writer,
      folly::CancellationToken cancellation) override {
    HttpResponse response{http::status::ok, request.version()};
    response.keep_alive(request.keep_alive());
    response.body() =
        std::string(request.target().data(), request.target().size());
    response.prepare_payload();
    (void)co_await writer.WriteResponse(std::move(response), cancellation);
  }
};

class ScopeGuard final : public RouteGuard {
 public:
  folly::coro::Task<Status> Check(const HttpRequest&,
                                  const RouteMetadata& metadata,
                                  RequestContext&,
                                  folly::CancellationToken) override {
    seen_scope = metadata.required_scope;
    if (deny) co_return absl::PermissionDeniedError("scope denied");
    co_return OkStatus();
  }
  bool deny = false;
  std::string seen_scope;
};

class FakeEmbeddingsService final
    : public ::inferx::server::handlers::EmbeddingsService {
 public:
  folly::coro::Task<StatusOr<::inferx::server::handlers::EmbeddingsOutput>>
  Embed(const ::inferx::api::EmbeddingsRequest& request,
        const ::inferx::server::model_registry::ModelRecord& model,
        const RequestContext& context, folly::CancellationToken) override {
    seen_model_version = model.version;
    seen_tenant = context.tenant_id;
    ::inferx::server::handlers::EmbeddingsOutput output;
    output.prompt_tokens = 7;
    for (size_t index = 0; index < request.input.size(); ++index) {
      output.values.push_back(
          {static_cast<float>(index), static_cast<float>(index + 1)});
    }
    co_return output;
  }

  std::string seen_model_version;
  std::string seen_tenant;
};

class FakeTokenizationService final
    : public ::inferx::server::tokenization::TokenizationService {
 public:
  StatusOr<::inferx::server::tokenization::TokenizedPrompt> TokenizeCompletion(
      const ::inferx::server::request::ModelVersion& model_version,
      std::string_view prompt, bool add_special_tokens) override {
    seen_version = model_version;
    seen_prompt = prompt;
    seen_add_special_tokens = add_special_tokens;
    return ::inferx::server::tokenization::TokenizedPrompt{
        model_version, {11, 12, 13}, 3};
  }

  StatusOr<::inferx::server::tokenization::TokenizedPrompt> TokenizeChat(
      const ::inferx::server::request::ModelVersion& model_version,
      const std::vector<::inferx::tokenizer::ChatMessage>& messages) override {
    seen_version = model_version;
    seen_messages = messages.size();
    return ::inferx::server::tokenization::TokenizedPrompt{
        model_version, {21, 22, 23}, 3};
  }

  StatusOr<std::string> Detokenize(
      const ::inferx::server::request::ModelVersion& model_version,
      const std::vector<::inferx::tokenizer::TokenId>& ids) override {
    seen_version = model_version;
    seen_detokenize_ids = ids;
    // Inverts TokenizeCompletion's fixed encoding so a tokenize/detokenize
    // round trip through the handlers is observable end to end.
    if (ids == std::vector<::inferx::tokenizer::TokenId>{11, 12, 13}) {
      return seen_prompt;
    }
    return absl::InvalidArgumentError("unknown token sequence");
  }

  std::string seen_version;
  std::string seen_prompt;
  bool seen_add_special_tokens = false;
  size_t seen_messages = 0;
  std::vector<::inferx::tokenizer::TokenId> seen_detokenize_ids;
};

class FakeRequestService final
    : public ::inferx::server::request::RequestService {
 public:
  folly::coro::Task<StatusOr<::inferx::server::scheduler_client::SubmitResult>>
  Submit(::inferx::server::scheduler_client::ScheduledRequest request,
         folly::CancellationToken) override {
    submitted = std::move(request);
    all_submitted.push_back(submitted);
    co_return ::inferx::server::scheduler_client::SubmitResult{
        submitted.request_id, 0};
  }

  folly::coro::AsyncGenerator<
      ::inferx::server::scheduler_client::GenerationEvent&&>
  Events(::inferx::server::request::RequestId request_id,
         folly::CancellationToken) override {
    if (emit_logprobs) {
      // One text-bearing token, then a token which completed no character:
      // the second event has an empty text_delta but still owes its report.
      ::inferx::server::scheduler_client::GenerationEvent first;
      first.request_id = request_id;
      first.sequence_number = 1;
      first.text_delta = "he";
      first.token_ids = {5};
      first.logprobs = {{5, -0.25f, {{5, -0.25f}, {6, -1.5f}}}};
      first.generated_tokens = 1;
      co_yield std::move(first);
      ::inferx::server::scheduler_client::GenerationEvent partial;
      partial.request_id = request_id;
      partial.sequence_number = 2;
      partial.token_ids = {7};
      partial.logprobs = {{7, -0.5f, {}}};
      partial.generated_tokens = 2;
      co_yield std::move(partial);
      ::inferx::server::scheduler_client::GenerationEvent terminal;
      terminal.request_id = request_id;
      terminal.sequence_number = 3;
      terminal.terminal = true;
      terminal.finish_reason =
          ::inferx::server::scheduler_client::FinishReason::kStop;
      terminal.usage = {.prompt_tokens = 3, .completion_tokens = 2};
      co_yield std::move(terminal);
      co_return;
    }
    ::inferx::server::scheduler_client::GenerationEvent first;
    first.request_id = request_id;
    first.sequence_number = 1;
    first.text_delta = "hello";
    first.generated_tokens = 1;
    co_yield std::move(first);
    ::inferx::server::scheduler_client::GenerationEvent terminal;
    terminal.request_id = request_id;
    terminal.sequence_number = 2;
    terminal.terminal = true;
    terminal.finish_reason =
        ::inferx::server::scheduler_client::FinishReason::kStop;
    terminal.usage = {.prompt_tokens = 3, .completion_tokens = 1};
    co_yield std::move(terminal);
  }

  folly::coro::Task<Status> Cancel(
      ::inferx::server::request::RequestId,
      ::inferx::server::request::CancellationReason) override {
    cancelled = true;
    co_return OkStatus();
  }

  ::inferx::server::scheduler_client::ScheduledRequest submitted;
  std::vector<::inferx::server::scheduler_client::ScheduledRequest>
      all_submitted;
  bool emit_logprobs = false;
  bool cancelled = false;
};

class FakeSchedulerClient final
    : public ::inferx::server::scheduler_client::SchedulerClient {
 public:
  folly::coro::Task<StatusOr<::inferx::server::scheduler_client::SubmitResult>>
  Submit(::inferx::server::scheduler_client::ScheduledRequest request,
         folly::CancellationToken) override {
    submitted = request;
    co_return ::inferx::server::scheduler_client::SubmitResult{
        request.request_id, 0};
  }
  folly::coro::AsyncGenerator<
      ::inferx::server::scheduler_client::GenerationEvent&&>
  Events(::inferx::server::request::RequestId request_id,
         folly::CancellationToken) override {
    ::inferx::server::scheduler_client::GenerationEvent event;
    event.request_id = request_id;
    event.sequence_number = 1;
    event.terminal = true;
    event.finish_reason =
        ::inferx::server::scheduler_client::FinishReason::kStop;
    event.usage = {.prompt_tokens = 2, .completion_tokens = 1};
    co_yield std::move(event);
  }
  folly::coro::Task<Status> Cancel(
      ::inferx::server::request::RequestId,
      ::inferx::server::request::CancellationReason) override {
    ++cancel_count;
    co_return OkStatus();
  }
  folly::coro::Task<StatusOr<::inferx::server::request::RequestSnapshot>>
  GetStatus(::inferx::server::request::RequestId,
            folly::CancellationToken) override {
    co_return UnimplementedError("not used");
  }
  folly::coro::Task<Status> UpdatePriority(
      ::inferx::server::request::RequestId,
      ::inferx::server::scheduler_client::PriorityClass,
      folly::CancellationToken) override {
    co_return OkStatus();
  }

  ::inferx::server::scheduler_client::ScheduledRequest submitted;
  int cancel_count = 0;
};

class FakeMetricsSource final
    : public ::inferx::server::observability::MetricsSource {
 public:
  ::inferx::server::observability::MetricsSnapshot Snapshot() const override {
    return {.running = 2,
            .waiting = 1,
            .blocks_in_use = 3,
            .blocks_total = 8,
            .steps = 9,
            .tokens_generated = 10,
            .last_step_ms = 4.5,
            .preemptions = 1,
            .cached_blocks = 2,
            .prefix_hit_tokens = 11,
            .prefix_miss_tokens = 12,
            .evicted_blocks = 13};
  }
};

class FakeInProcessGeneration final
    : public ::inferx::server::scheduler_client::InProcessGeneration {
 public:
  folly::coro::Task<StatusOr<std::optional<
      ::inferx::server::scheduler_client::InProcessGenerationEvent>>>
  Next(folly::CancellationToken) override {
    next_thread = std::this_thread::get_id();
    ::inferx::server::scheduler_client::InProcessGenerationEvent event;
    if (index == 0) {
      event = {.text = "delta", .generated_tokens = 1};
    } else if (index == 1) {
      event = {.terminal = true,
               .finish_reason =
                   ::inferx::server::scheduler_client::FinishReason::kStop,
               .generated_tokens = 1};
    } else {
      co_return std::optional<
          ::inferx::server::scheduler_client::InProcessGenerationEvent>{};
    }
    ++index;
    co_return std::optional<
        ::inferx::server::scheduler_client::InProcessGenerationEvent>(
        std::move(event));
  }
  void Cancel() override { cancelled = true; }
  uint32_t prompt_tokens() const override { return 3; }
  size_t index = 0;
  bool cancelled = false;
  std::thread::id next_thread;
};

class FakeInProcessBackend final
    : public ::inferx::server::scheduler_client::InProcessEngineBackend {
 public:
  StatusOr<
      std::shared_ptr<::inferx::server::scheduler_client::InProcessGeneration>>
  Submit(const ::inferx::server::scheduler_client::ScheduledRequest& request)
      override {
    seen_version = request.model_version;
    generation = std::make_shared<FakeInProcessGeneration>();
    return std::static_pointer_cast<
        ::inferx::server::scheduler_client::InProcessGeneration>(generation);
  }
  std::string seen_version;
  std::shared_ptr<FakeInProcessGeneration> generation;
};

TEST(BeastFollyAdapterTest, CompletesAndReturnsToFollyExecutor) {
  asio::io_context io;
  auto work = asio::make_work_guard(io);
  std::thread io_thread([&] { io.run(); });
  folly::CPUThreadPoolExecutor executor(1);

  auto task = [&]() -> folly::coro::Task<StatusOr<size_t>> {
    co_return co_await AwaitAsio<size_t>(
        io.get_executor(), [](auto completion) { completion({}, 17); }, [] {});
  };
  auto future = folly::coro::co_withExecutor(&executor, task()).start();
  StatusOr<size_t> result = std::move(future).get();
  ASSERT_TRUE(result.ok()) << result.status();
  EXPECT_EQ(*result, 17);

  work.reset();
  io.stop();
  io_thread.join();
}

TEST(BeastFollyAdapterTest, CancellationWinsBeforeInitiation) {
  asio::io_context io;
  auto work = asio::make_work_guard(io);
  std::thread io_thread([&] { io.run(); });
  folly::CPUThreadPoolExecutor executor(1);
  folly::CancellationSource cancellation;
  cancellation.requestCancellation();

  auto task = [&]() -> folly::coro::Task<StatusOr<size_t>> {
    co_return co_await AwaitAsio<size_t>(
        io.get_executor(), [](auto) {}, [] {}, cancellation.getToken());
  };
  auto future = folly::coro::co_withExecutor(&executor, task()).start();
  StatusOr<size_t> result = std::move(future).get();
  EXPECT_TRUE(absl::IsCancelled(result.status()));

  work.reset();
  io.stop();
  io_thread.join();
}

TEST(RoutesTest, DistinguishesNotFoundAndMethodNotAllowed) {
  Routes routes;
  ASSERT_TRUE(
      routes.Add(http::verb::get, "/ready", std::make_shared<OkHandler>())
          .ok());

  CapturingWriter writer;
  HttpRequest wrong_method{http::verb::post, "/ready", 11};
  folly::coro::blockingWait(
      routes.Handle(std::move(wrong_method), {}, writer, {}));
  EXPECT_EQ(writer.status, 405);

  CapturingWriter missing_writer;
  HttpRequest missing{http::verb::get, "/missing", 11};
  folly::coro::blockingWait(
      routes.Handle(std::move(missing), {}, missing_writer, {}));
  EXPECT_EQ(missing_writer.status, 404);
}

TEST(RoutesTest, AppliesMetadataBodyLimitAndGuardBeforeHandler) {
  auto guard = std::make_shared<ScopeGuard>();
  Routes routes(guard);
  ASSERT_TRUE(routes
                  .Add(http::verb::post, "/completion",
                       {.name = "completion",
                        .max_body_bytes = 4,
                        .authentication_required = true,
                        .required_scope = "inference.invoke"},
                       std::make_shared<OkHandler>())
                  .ok());

  CapturingWriter oversized_writer;
  HttpRequest oversized{http::verb::post, "/completion", 11};
  oversized.body() = "12345";
  folly::coro::blockingWait(
      routes.Handle(std::move(oversized), {}, oversized_writer, {}));
  EXPECT_EQ(oversized_writer.status, 413);
  EXPECT_TRUE(guard->seen_scope.empty());

  guard->deny = true;
  CapturingWriter denied_writer;
  HttpRequest denied{http::verb::post, "/completion", 11};
  denied.body() = "1234";
  folly::coro::blockingWait(
      routes.Handle(std::move(denied), {}, denied_writer, {}));
  EXPECT_EQ(denied_writer.status, 403);
  EXPECT_EQ(guard->seen_scope, "inference.invoke");
}

TEST(RoutesTest, PrefixRoutesYieldToExactRoutesAndBoundTheirMatch) {
  Routes routes;
  ASSERT_TRUE(
      routes.Add(http::verb::get, "/v1/models", std::make_shared<OkHandler>())
          .ok());
  ASSERT_TRUE(routes
                  .AddPrefix(http::verb::get, "/v1/models/",
                             {.name = "models.retrieve",
                              .max_body_bytes = 1,
                              .authentication_required = false},
                             std::make_shared<TargetEchoHandler>())
                  .ok());
  // The trailing slash is the parameter boundary; without it a prefix could
  // shadow exact siblings.
  EXPECT_FALSE(routes
                   .AddPrefix(http::verb::get, "/no-trailing-slash",
                              {.authentication_required = false},
                              std::make_shared<TargetEchoHandler>())
                   .ok());

  CapturingWriter exact_writer;
  HttpRequest exact{http::verb::get, "/v1/models", 11};
  folly::coro::blockingWait(routes.Handle(std::move(exact), {}, exact_writer, {}));
  EXPECT_EQ(exact_writer.status, 200);
  EXPECT_EQ(exact_writer.body, "{\"ok\":true}");

  CapturingWriter id_writer;
  HttpRequest by_id{http::verb::get, "/v1/models/llama", 11};
  folly::coro::blockingWait(routes.Handle(std::move(by_id), {}, id_writer, {}));
  EXPECT_EQ(id_writer.status, 200);
  EXPECT_EQ(id_writer.body, "/v1/models/llama");

  CapturingWriter missing_writer;
  HttpRequest missing{http::verb::get, "/v1/modelsllama", 11};
  folly::coro::blockingWait(
      routes.Handle(std::move(missing), {}, missing_writer, {}));
  EXPECT_EQ(missing_writer.status, 404);

  // A prefix path with the wrong verb is a known path: 405, not 404.
  CapturingWriter method_writer;
  HttpRequest wrong_method{http::verb::post, "/v1/models/llama", 11};
  folly::coro::blockingWait(
      routes.Handle(std::move(wrong_method), {}, method_writer, {}));
  EXPECT_EQ(method_writer.status, 405);
}

TEST(SseWriterTest, FramesEventsCommentsAndFinishes) {
  CapturingWriter writer;
  SseWriter sse(&writer);
  EXPECT_TRUE(folly::coro::blockingWait(sse.Start(11, true, "req_1", {})).ok());
  EXPECT_TRUE(folly::coro::blockingWait(sse.Event("one\ntwo", {})).ok());
  EXPECT_TRUE(folly::coro::blockingWait(sse.Comment("keepalive", {})).ok());
  EXPECT_TRUE(folly::coro::blockingWait(sse.Finish({})).ok());
  EXPECT_EQ(writer.body, "data: one\ndata: two\n\n: keepalive\n\n");
  EXPECT_TRUE(writer.finished);
}

TEST(ErrorMappingTest, MapsRetryableStatusAndEscapesFields) {
  const ::inferx::server::api::HttpError error =
      ::inferx::server::api::MapStatus(ResourceExhaustedError("quota \"full\""),
                                       "max_tokens");
  EXPECT_EQ(error.status, 429);
  EXPECT_EQ(error.type, "rate_limit_error");
  EXPECT_EQ(error.retry_after, "1");
  EXPECT_EQ(error.Json("req_1"),
            "{\"error\":{\"message\":\"quota \\\"full\\\"\","
            "\"type\":\"rate_limit_error\",\"param\":\"max_tokens\","
            "\"code\":\"capacity_exceeded\",\"request_id\":\"req_1\"}}");
}

TEST(OpenAiProtocolTest, ValidatesExtendedSamplingParameters) {
  auto parsed = ::inferx::api::ParseCompletionRequest(
      R"({"model":"m","prompt":"p","stream":true,"stream_options":{"include_usage":true},"top_k":0,"repetition_penalty":1,"n":1,"seed":42})");
  ASSERT_TRUE(parsed.ok()) << parsed.status();
  EXPECT_EQ(parsed->sampling.top_k, 0);
  EXPECT_FLOAT_EQ(parsed->sampling.repetition_penalty, 1.0f);
  EXPECT_TRUE(parsed->sampling.include_usage);

  EXPECT_FALSE(::inferx::api::ParseCompletionRequest(
                   R"({"model":"m","prompt":"p","stream_options":{}})")
                   .ok());
  // Implemented since the vLLM-compat work: n>1, top_k, and
  // repetition_penalty parse into the sampling request rather than erroring.
  auto multi = ::inferx::api::ParseCompletionRequest(
      R"({"model":"m","prompt":"p","n":2})");
  ASSERT_TRUE(multi.ok()) << multi.status();
  EXPECT_EQ(multi->sampling.n, 2);
  auto top_k = ::inferx::api::ParseCompletionRequest(
      R"({"model":"m","prompt":"p","top_k":40})");
  ASSERT_TRUE(top_k.ok()) << top_k.status();
  EXPECT_EQ(top_k->sampling.top_k, 40);
  auto penalty = ::inferx::api::ParseCompletionRequest(
      R"({"model":"m","prompt":"p","repetition_penalty":1.1})");
  ASSERT_TRUE(penalty.ok()) << penalty.status();
  EXPECT_FLOAT_EQ(penalty->sampling.repetition_penalty, 1.1f);
  EXPECT_FALSE(::inferx::api::ParseCompletionRequest(
                   R"({"model":"m","prompt":"p","seed":-1})")
                   .ok());
  EXPECT_FALSE(::inferx::api::ParseCompletionRequest(
                   R"({"model":"m","prompt":"p","stop":[""]})")
                   .ok());
}

TEST(OpenAiProtocolTest, ParsesEmbeddingInputsWithoutExecutingThem) {
  auto parsed = ::inferx::api::ParseEmbeddingsRequest(
      R"({"model":"embed","input":["one","two"],"encoding_format":"float","dimensions":128})");
  ASSERT_TRUE(parsed.ok()) << parsed.status();
  EXPECT_EQ(parsed->input.size(), 2);
  EXPECT_TRUE(parsed->has_dimensions);
  EXPECT_EQ(parsed->dimensions, 128);
  EXPECT_FALSE(::inferx::api::ParseEmbeddingsRequest(
                   R"({"model":"embed","input":[],"encoding_format":"hex"})")
                   .ok());
}

TEST(RequestStateTest, EnforcesLifecycleAndIdempotentCancellation) {
  using ::inferx::server::request::RequestContext;
  using ::inferx::server::request::RequestState;
  RequestContext context;
  EXPECT_TRUE(context.TransitionTo(RequestState::kValidating));
  EXPECT_TRUE(context.TransitionTo(RequestState::kAuthenticated));
  EXPECT_TRUE(context.TransitionTo(RequestState::kAdmissionPending));
  EXPECT_TRUE(context.TransitionTo(RequestState::kQueued));
  EXPECT_TRUE(context.TransitionTo(RequestState::kPrefilling));
  EXPECT_TRUE(context.TransitionTo(RequestState::kDecoding));
  EXPECT_TRUE(context.TransitionTo(RequestState::kFinishing));
  EXPECT_TRUE(context.TransitionTo(RequestState::kCompleted));
  EXPECT_FALSE(context.TransitionTo(RequestState::kDecoding));

  RequestContext cancelled;
  cancelled.Cancel(
      ::inferx::server::request::CancellationReason::kClientDisconnected);
  cancelled.Cancel(::inferx::server::request::CancellationReason::kInternal);
  EXPECT_EQ(cancelled.state, RequestState::kCancelled);
  EXPECT_TRUE(cancelled.cancellation.isCancellationRequested());
}

TEST(DeadlineTest, ReturnsResultOrCancelsExpiredOperation) {
  folly::CancellationSource success_cancellation;
  folly::ManualTimekeeper timekeeper;
  auto success = []() -> folly::coro::Task<StatusOr<int>> { co_return 7; };
  auto result =
      folly::coro::blockingWait(::inferx::server::coroutine::WithDeadline(
          success(), std::chrono::steady_clock::now() + std::chrono::seconds(1),
          success_cancellation, &timekeeper));
  ASSERT_TRUE(result.ok()) << result.status();
  EXPECT_EQ(*result, 7);
  EXPECT_FALSE(success_cancellation.isCancellationRequested());

  bool started = false;
  folly::CancellationSource expired_cancellation;
  auto expired = [&]() -> folly::coro::Task<StatusOr<int>> {
    started = true;
    co_return 9;
  };
  result = folly::coro::blockingWait(::inferx::server::coroutine::WithDeadline(
      expired(),
      std::chrono::steady_clock::now() - std::chrono::milliseconds(1),
      expired_cancellation));
  EXPECT_EQ(result.status().code(), absl::StatusCode::kDeadlineExceeded);
  EXPECT_TRUE(expired_cancellation.isCancellationRequested());
  EXPECT_FALSE(started);
}

TEST(RequestManagerTest, OwnsSnapshotsAndIdempotentTerminalCleanup) {
  ::inferx::server::request::RequestManager manager;
  ::inferx::server::request::RequestContext context;
  context.request_id = "req_manager";
  auto created = manager.Create(std::move(context));
  ASSERT_TRUE(created.ok()) << created.status();
  EXPECT_EQ(manager.active_count(), 1);
  ASSERT_TRUE(manager
                  .Cancel("req_manager",
                          ::inferx::server::request::CancellationReason::
                              kClientDisconnected)
                  .ok());
  auto snapshot = manager.GetStatus("req_manager");
  ASSERT_TRUE(snapshot.ok()) << snapshot.status();
  EXPECT_EQ(snapshot->state,
            ::inferx::server::request::RequestState::kCancelled);
  EXPECT_TRUE(manager.Finalize("req_manager").ok());
  EXPECT_TRUE(manager.Finalize("req_manager").ok());
  EXPECT_EQ(manager.active_count(), 0);
  EXPECT_EQ(manager.completed_count(), 1);
  EXPECT_TRUE(manager.GetStatus("req_manager").ok());
}

TEST(RequestIdTest, GeneratesPrefixedSortableUuidV7) {
  using namespace std::chrono_literals;
  const auto epoch = std::chrono::system_clock::time_point(1234567890ms);
  const std::string first = ::inferx::server::request::GenerateRequestId(epoch);
  const std::string later =
      ::inferx::server::request::GenerateRequestId(epoch + 1ms);
  EXPECT_EQ(first.size(), 40);
  EXPECT_EQ(first.substr(0, 4), "req_");
  EXPECT_EQ(first[18], '7');
  EXPECT_TRUE(first[23] == '8' || first[23] == '9' || first[23] == 'a' ||
              first[23] == 'b');
  EXPECT_LT(first, later);
}

TEST(RequestManagerTest, BoundsAndExpiresCompletedSnapshots) {
  using Clock = std::chrono::steady_clock;
  auto now = Clock::time_point(std::chrono::seconds(10));
  ::inferx::server::request::RequestManager manager(
      {.completed_retention = std::chrono::seconds(5),
       .max_completed_snapshots = 1,
       .now = [&] { return now; }});
  for (const std::string id : {"req_one", "req_two"}) {
    ::inferx::server::request::RequestContext context;
    context.request_id = id;
    auto created = manager.Create(std::move(context));
    ASSERT_TRUE(created.ok()) << created.status();
    ASSERT_TRUE(
        manager
            .Cancel(
                id,
                ::inferx::server::request::CancellationReason::kAdministrative)
            .ok());
    ASSERT_TRUE(manager.Finalize(id).ok());
  }
  EXPECT_EQ(manager.completed_count(), 1);
  EXPECT_FALSE(manager.GetStatus("req_one").ok());
  now += std::chrono::seconds(5);
  EXPECT_EQ(manager.completed_count(), 0);
}

TEST(EventBufferTest, BoundsAndDeliversEvents) {
  auto result = ::inferx::server::streaming::EventBuffer::Create(1);
  ASSERT_TRUE(result.ok()) << result.status();
  auto buffer = std::move(*result);
  ::inferx::server::scheduler_client::GenerationEvent first;
  first.sequence_number = 1;
  EXPECT_EQ(buffer->TryPush(first),
            ::inferx::server::streaming::PushResult::kQueued);
  ::inferx::server::scheduler_client::GenerationEvent second;
  second.sequence_number = 2;
  EXPECT_EQ(buffer->TryPush(second),
            ::inferx::server::streaming::PushResult::kFull);
  auto event = folly::coro::blockingWait(buffer->Next());
  ASSERT_TRUE(event.has_value());
  EXPECT_EQ(event->sequence_number, 1);
  EXPECT_EQ(buffer->TryPush(second),
            ::inferx::server::streaming::PushResult::kQueued);
  buffer->Close();
  EXPECT_EQ(buffer->TryPush(first),
            ::inferx::server::streaming::PushResult::kClosed);
}

TEST(EventBufferTest, EnforcesByteBudgetAndAccountsConsumption) {
  auto result = ::inferx::server::streaming::EventBuffer::Create(4, 128);
  ASSERT_TRUE(result.ok()) << result.status();
  auto buffer = std::move(*result);
  ::inferx::server::scheduler_client::GenerationEvent event;
  event.text_delta = std::string(100, 'x');
  EXPECT_EQ(buffer->TryPush(event),
            ::inferx::server::streaming::PushResult::kQueued);
  EXPECT_GT(buffer->queued_bytes(), 0u);
  event.text_delta = "y";
  EXPECT_EQ(buffer->TryPush(event),
            ::inferx::server::streaming::PushResult::kFull);
  auto consumed = folly::coro::blockingWait(buffer->Next());
  ASSERT_TRUE(consumed.has_value());
  EXPECT_EQ(buffer->queued_bytes(), 0u);
}

TEST(BackpressureControllerTest, CancelsOnlyAfterContinuousFullPeriod) {
  using Clock = std::chrono::steady_clock;
  const auto start = Clock::time_point(std::chrono::seconds(10));
  ::inferx::server::streaming::BackpressureController controller(
      std::chrono::milliseconds(100));
  controller.Observe(::inferx::server::streaming::PushResult::kFull, start);
  EXPECT_FALSE(controller.ShouldCancel(start + std::chrono::milliseconds(99)));
  EXPECT_TRUE(controller.ShouldCancel(start + std::chrono::milliseconds(100)));
  controller.Observe(::inferx::server::streaming::PushResult::kQueued,
                     start + std::chrono::milliseconds(101));
  EXPECT_FALSE(controller.full());
}

TEST(EventRouterTest, RejectsGapsAndWrongRequests) {
  auto result = ::inferx::server::streaming::EventBuffer::Create(2);
  ASSERT_TRUE(result.ok()) << result.status();
  auto buffer = std::move(*result);
  ::inferx::server::streaming::EventRouter router(buffer.get(), "req_router");

  ::inferx::server::scheduler_client::GenerationEvent wrong;
  wrong.request_id = "other";
  wrong.sequence_number = 1;
  EXPECT_FALSE(router.Route(std::move(wrong)).ok());

  ::inferx::server::scheduler_client::GenerationEvent gap;
  gap.request_id = "req_router";
  gap.sequence_number = 2;
  EXPECT_FALSE(router.Route(std::move(gap)).ok());

  ::inferx::server::scheduler_client::GenerationEvent first;
  first.request_id = "req_router";
  first.sequence_number = 1;
  EXPECT_TRUE(router.Route(std::move(first)).ok());
  EXPECT_EQ(router.next_sequence(), 2);
}

TEST(AdmissionTest, EnforcesAndReleasesTenantReservations) {
  ::inferx::server::admission::AdmissionController controller(
      {.max_active_requests = 2,
       .max_reserved_tokens = 100,
       .max_active_per_tenant = 1});
  const ::inferx::server::admission::AdmissionRequest first{"tenant_a", 40};
  EXPECT_TRUE(controller.TryAdmit(first).admitted());
  EXPECT_FALSE(controller.TryAdmit(first).admitted());
  const ::inferx::server::admission::AdmissionRequest other{"tenant_b", 70};
  EXPECT_FALSE(controller.TryAdmit(other).admitted());
  controller.Release(first);
  EXPECT_TRUE(controller.TryAdmit(other).admitted());
  EXPECT_EQ(controller.active_requests(), 1);
  EXPECT_EQ(controller.reserved_tokens(), 70);
  controller.Release(other);
  EXPECT_EQ(controller.active_requests(), 0);
}

TEST(AdmissionTest, ReservationReleaseIsIdempotent) {
  ::inferx::server::admission::AdmissionController controller(
      {.max_active_requests = 2,
       .max_reserved_tokens = 100,
       .max_active_per_tenant = 2});
  const ::inferx::server::admission::AdmissionRequest request{"tenant", 20};
  const auto first = controller.TryAdmit(request);
  const auto second = controller.TryAdmit(request);
  ASSERT_TRUE(first.admitted());
  ASSERT_TRUE(second.admitted());
  ASSERT_NE(first.reservation_id, second.reservation_id);

  controller.Release(first.reservation_id);
  controller.Release(first.reservation_id);
  EXPECT_EQ(controller.active_requests(), 1);
  EXPECT_EQ(controller.reserved_tokens(), 20);

  controller.Release(second.reservation_id);
  EXPECT_EQ(controller.active_requests(), 0);
  EXPECT_EQ(controller.reserved_tokens(), 0);
}

TEST(CapacityPolicyTest, RejectsUnavailableAndExhaustedCapacity) {
  using ::inferx::server::admission::AdmissionReason;
  using ::inferx::server::admission::CapacityPolicy;
  using ::inferx::server::admission::CapacitySnapshot;
  CapacityPolicy policy(
      {.max_queued_requests = 2,
       .max_event_buffer_bytes = 100,
       .overload_retry_after = std::chrono::milliseconds(250),
       .unavailable_retry_after = std::chrono::milliseconds(500)});

  auto decision = policy.Evaluate({});
  EXPECT_EQ(decision.reason, AdmissionReason::kSchedulerUnavailable);
  EXPECT_EQ(decision.retry_after, std::chrono::milliseconds(500));

  decision = policy.Evaluate({.scheduler_healthy = true});
  EXPECT_EQ(decision.reason, AdmissionReason::kModelUnavailable);

  decision = policy.Evaluate(
      {.scheduler_healthy = true, .model_ready = true, .queued_requests = 2});
  EXPECT_EQ(decision.reason, AdmissionReason::kQueueCapacity);
  EXPECT_EQ(decision.retry_after, std::chrono::milliseconds(250));

  decision = policy.Evaluate({.scheduler_healthy = true,
                              .model_ready = true,
                              .event_buffer_bytes = 100});
  EXPECT_EQ(decision.reason, AdmissionReason::kEventBufferCapacity);

  decision = policy.Evaluate({.scheduler_healthy = true,
                              .model_ready = true,
                              .queued_requests = 1,
                              .event_buffer_bytes = 99});
  EXPECT_TRUE(decision.admitted());
  EXPECT_FALSE(decision.retryable());
}

TEST(RateLimiterTest, RefillsAndReportsRetryAdvice) {
  using Clock = std::chrono::steady_clock;
  const auto start = Clock::time_point(std::chrono::seconds(20));
  ::inferx::server::admission::RateLimiter limiter({1, 1, 10, 10});
  EXPECT_TRUE(limiter.Consume("tenant_rate", 10, start).allowed);
  auto rejected = limiter.Consume("tenant_rate", 1, start);
  EXPECT_FALSE(rejected.allowed);
  EXPECT_GT(rejected.retry_after.count(), 0);
  EXPECT_TRUE(limiter.Consume("tenant_rate", 1, start + std::chrono::seconds(1))
                  .allowed);
}

TEST(RateLimiterTest, ReconcilesReservedTokensAgainstActualUsage) {
  using Clock = std::chrono::steady_clock;
  const auto start = Clock::time_point(std::chrono::seconds(30));
  ::inferx::server::admission::RateLimiter limiter({10, 1, 100, 1});

  ASSERT_TRUE(limiter.Consume("tenant_reconcile", 80, start).allowed);
  EXPECT_FALSE(limiter.Consume("tenant_reconcile", 30, start).allowed);

  ASSERT_TRUE(limiter.Reconcile("tenant_reconcile", 80, 20, start).ok());
  EXPECT_TRUE(limiter.Consume("tenant_reconcile", 30, start).allowed);

  ASSERT_TRUE(limiter.Reconcile("tenant_reconcile", 30, 50, start).ok());
  EXPECT_FALSE(limiter.Consume("tenant_reconcile", 31, start).allowed);
  EXPECT_EQ(limiter.Reconcile("unknown", 1, 1, start).code(),
            absl::StatusCode::kFailedPrecondition);
}

TEST(RateLimiterTest, EnforcesTenantAndApiKeyBucketsAtomically) {
  using Decision = ::inferx::server::admission::RateLimitDecision;
  using Clock = std::chrono::steady_clock;
  const auto start = Clock::time_point(std::chrono::seconds(40));
  ::inferx::server::admission::RateLimiter limiter(
      {.request_capacity = 3,
       .request_refill_per_second = 1,
       .token_capacity = 100,
       .token_refill_per_second = 1,
       .api_key_request_capacity = 1,
       .api_key_request_refill_per_second = 1,
       .api_key_token_capacity = 100,
       .api_key_token_refill_per_second = 1});

  EXPECT_TRUE(limiter.Consume("tenant", "key_a", 10, start).allowed);
  auto key_rejection = limiter.Consume("tenant", "key_a", 10, start);
  EXPECT_FALSE(key_rejection.allowed);
  EXPECT_EQ(key_rejection.reason, Decision::Reason::kApiKeyLimit);

  // The rejected key_a attempt must not consume tenant capacity, leaving room
  // for both other keys.
  EXPECT_TRUE(limiter.Consume("tenant", "key_b", 10, start).allowed);
  EXPECT_TRUE(limiter.Consume("tenant", "key_c", 10, start).allowed);
  auto tenant_rejection = limiter.Consume("tenant", "key_d", 10, start);
  EXPECT_FALSE(tenant_rejection.allowed);
  EXPECT_EQ(tenant_rejection.reason, Decision::Reason::kTenantLimit);

  ASSERT_TRUE(limiter.Reconcile("tenant", "key_a", 10, 5, start).ok());
  EXPECT_EQ(limiter.Reconcile("tenant", "missing", 10, 5, start).code(),
            absl::StatusCode::kFailedPrecondition);
}

TEST(AuthTest, StoresHashesAndEnforcesScopes) {
  ::inferx::server::auth::ApiKeyStore store;
  ::inferx::server::auth::Principal principal;
  principal.tenant_id = "tenant_a";
  principal.subject = "user_a";
  principal.scopes.insert("completions:write");
  ASSERT_TRUE(store.AddHash("abc123", principal).ok());
  auto found = store.LookupHash("abc123");
  ASSERT_TRUE(found.ok()) << found.status();
  EXPECT_EQ(found->tenant_id, "tenant_a");
  EXPECT_TRUE(
      ::inferx::server::auth::Authorize(*found, "completions:write").ok());
  EXPECT_EQ(::inferx::server::auth::Authorize(*found, "models:admin").code(),
            absl::StatusCode::kPermissionDenied);
  EXPECT_EQ(store.LookupHash("missing").status().code(),
            absl::StatusCode::kUnauthenticated);
}

TEST(AuthTest, RotatesCompleteKeySnapshotAtomically) {
  using ::inferx::server::auth::ApiKeyRecord;
  using ::inferx::server::auth::ApiKeyStore;
  using ::inferx::server::auth::Principal;
  ApiKeyStore store;
  ASSERT_TRUE(store.AddHash("old", Principal{"tenant", "old_user"}).ok());

  std::vector<ApiKeyRecord> replacement{
      {"new_a", Principal{"tenant", "user_a"}},
      {"new_b", Principal{"tenant", "user_b"}}};
  ASSERT_TRUE(store.ReplaceAll(std::move(replacement)).ok());
  EXPECT_EQ(store.size(), 2);
  EXPECT_EQ(store.LookupHash("old").status().code(),
            absl::StatusCode::kUnauthenticated);
  EXPECT_EQ(store.LookupHash("new_a")->subject, "user_a");

  std::vector<ApiKeyRecord> invalid{
      {"duplicate", Principal{"tenant", "first"}},
      {"duplicate", Principal{"tenant", "second"}}};
  EXPECT_EQ(store.ReplaceAll(std::move(invalid)).code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(store.size(), 2);
  EXPECT_TRUE(store.LookupHash("new_b").ok());
}

TEST(AuthTest, AuthenticatesBearerTokenWithoutStoringRawCredential) {
  using ::inferx::server::auth::ApiKeyAuthenticator;
  using ::inferx::server::auth::ApiKeyStore;
  using ::inferx::server::auth::Principal;
  ApiKeyStore store;
  ASSERT_TRUE(store
                  .AddHash("2bb80d537b1da3e38bd30361aa855686bde0eacd7162fef6a25"
                           "fe97bf527a25b",
                           Principal{"tenant", "user", "key_1"})
                  .ok());
  ApiKeyAuthenticator authenticator(&store);

  auto principal = authenticator.Authenticate("Bearer secret");
  ASSERT_TRUE(principal.ok()) << principal.status();
  EXPECT_EQ(principal->key_id, "key_1");
  EXPECT_EQ(store.size(), 1);
  EXPECT_FALSE(store.LookupHash("secret").ok());
  EXPECT_EQ(authenticator.Authenticate("").status().code(),
            absl::StatusCode::kUnauthenticated);
  EXPECT_EQ(authenticator.Authenticate("Basic secret").status().code(),
            absl::StatusCode::kUnauthenticated);
  EXPECT_EQ(authenticator.Authenticate("Bearer wrong token").status().code(),
            absl::StatusCode::kUnauthenticated);
}

TEST(AuthTest, DevelopmentBypassMustBeExplicit) {
  ::inferx::server::auth::ApiKeyStore store;
  ::inferx::server::auth::ApiKeyAuthenticator secure(&store);
  EXPECT_EQ(secure.Authenticate("").status().code(),
            absl::StatusCode::kUnauthenticated);
  ::inferx::server::auth::ApiKeyAuthenticator development(&store, true);
  auto principal = development.Authenticate("");
  ASSERT_TRUE(principal.ok()) << principal.status();
  EXPECT_EQ(principal->tenant_id, "development");
}

TEST(AuthMiddlewareTest, EnforcesPerRouteAuthenticationAndScope) {
  auto store = std::make_shared<::inferx::server::auth::ApiKeyStore>();
  ::inferx::server::auth::Principal principal{"tenant", "user", "key"};
  principal.scopes.insert("inference.invoke");
  ASSERT_TRUE(store
                  ->AddHash("2bb80d537b1da3e38bd30361aa855686bde0eacd7162fef6a2"
                            "5fe97bf527a25b",
                            principal)
                  .ok());
  auto authenticator =
      std::make_shared<::inferx::server::auth::ApiKeyAuthenticator>(
          store.get());
  auto guard = std::make_shared<::inferx::server::middleware::BearerRouteGuard>(
      authenticator);
  Routes routes(guard);
  auto invoke_handler = std::make_shared<OkHandler>();
  ASSERT_TRUE(routes
                  .Add(http::verb::post, "/invoke",
                       {.authentication_required = true,
                        .required_scope = "inference.invoke"},
                       invoke_handler)
                  .ok());
  ASSERT_TRUE(routes
                  .Add(http::verb::get, "/health",
                       {.authentication_required = false},
                       std::make_shared<OkHandler>())
                  .ok());

  CapturingWriter missing_writer;
  HttpRequest missing{http::verb::post, "/invoke", 11};
  folly::coro::blockingWait(
      routes.Handle(std::move(missing), {}, missing_writer, {}));
  EXPECT_EQ(missing_writer.status, 401);

  CapturingWriter allowed_writer;
  HttpRequest allowed{http::verb::post, "/invoke", 11};
  allowed.set(http::field::authorization, "Bearer secret");
  folly::coro::blockingWait(
      routes.Handle(std::move(allowed), {}, allowed_writer, {}));
  EXPECT_EQ(allowed_writer.status, 200);
  EXPECT_TRUE(invoke_handler->last_context.authenticated);
  EXPECT_EQ(invoke_handler->last_context.tenant_id, "tenant");
  EXPECT_EQ(invoke_handler->last_context.subject, "user");
  EXPECT_EQ(invoke_handler->last_context.api_key_id, "key");
  EXPECT_TRUE(invoke_handler->last_context.scopes.contains("inference.invoke"));

  CapturingWriter health_writer;
  HttpRequest health{http::verb::get, "/health", 11};
  folly::coro::blockingWait(
      routes.Handle(std::move(health), {}, health_writer, {}));
  EXPECT_EQ(health_writer.status, 200);
}

TEST(AuthMiddlewareTest, DevelopmentBypassAuthorizesConfiguredApiScopes) {
  auto store = std::make_shared<::inferx::server::auth::ApiKeyStore>();
  auto authenticator =
      std::make_shared<::inferx::server::auth::ApiKeyAuthenticator>(store.get(),
                                                                    true);
  auto guard = std::make_shared<::inferx::server::middleware::BearerRouteGuard>(
      authenticator);
  Routes routes(guard);
  auto handler = std::make_shared<OkHandler>();
  ASSERT_TRUE(routes
                  .Add(http::verb::post, "/invoke",
                       {.authentication_required = true,
                        .required_scope = "inference.invoke"},
                       handler)
                  .ok());

  CapturingWriter writer;
  HttpRequest request{http::verb::post, "/invoke", 11};
  folly::coro::blockingWait(routes.Handle(std::move(request), {}, writer, {}));
  EXPECT_EQ(writer.status, 200);
  EXPECT_TRUE(handler->last_context.authenticated);
  EXPECT_EQ(handler->last_context.tenant_id, "development");
}

TEST(ModelRegistryTest, ResolvesReadyAliasOnly) {
  ::inferx::server::model_registry::Registry registry;
  ::inferx::server::model_registry::ModelRecord record;
  record.id = "model";
  record.version = "v1";
  record.alias = "current";
  record.state = ::inferx::server::model_registry::ModelState::kLoading;
  ASSERT_TRUE(registry.Register(record).ok());
  EXPECT_FALSE(registry.Resolve("current").ok());
  ASSERT_TRUE(
      registry
          .SetState("model", "v1",
                    ::inferx::server::model_registry::ModelState::kWarming)
          .ok());
  ASSERT_TRUE(
      registry
          .SetState("model", "v1",
                    ::inferx::server::model_registry::ModelState::kReady)
          .ok());
  auto resolved = registry.Resolve("current");
  ASSERT_TRUE(resolved.ok()) << resolved.status();
  EXPECT_EQ(resolved->version, "v1");
  EXPECT_EQ(registry.ReadyModels().size(), 1);
}

TEST(ModelRegistryTest, EnforcesLifecycleCapabilitiesAndTenantVisibility) {
  using namespace ::inferx::server::model_registry;
  Registry registry;
  ModelRecord record;
  record.id = "embedding";
  record.version = "v2";
  record.alias = "embedding-current";
  record.supports_generation = false;
  record.supports_embeddings = true;
  record.embedding_dimensions = 768;
  record.visible_tenants.insert("tenant_a");
  ASSERT_TRUE(registry.Register(record).ok());
  EXPECT_FALSE(registry.SetState("embedding", "v2", ModelState::kReady).ok());
  ASSERT_TRUE(registry.SetState("embedding", "v2", ModelState::kLoading).ok());
  ASSERT_TRUE(registry.SetState("embedding", "v2", ModelState::kWarming).ok());
  ASSERT_TRUE(registry.SetState("embedding", "v2", ModelState::kReady).ok());
  auto visible = registry.Resolve("embedding-current", "tenant_a");
  ASSERT_TRUE(visible.ok()) << visible.status();
  EXPECT_TRUE(visible->supports_embeddings);
  EXPECT_EQ(visible->embedding_dimensions, 768);
  EXPECT_FALSE(registry.Resolve("embedding-current", "tenant_b").ok());
  EXPECT_EQ(registry.ReadyModels("tenant_b").size(), 0);
}

TEST(TokenizationContractTest, CarriesImmutableModelVersionAndAccounting) {
  ::inferx::server::tokenization::TokenizedPrompt prompt;
  prompt.model_version = "model@v1";
  prompt.token_ids = {1, 2, 3};
  prompt.prompt_tokens = static_cast<uint32_t>(prompt.token_ids.size());
  EXPECT_EQ(prompt.model_version, "model@v1");
  EXPECT_EQ(prompt.prompt_tokens, 3);
}

TEST(SchedulerContractTest, CarriesTypedTerminalUsageAndEmbeddings) {
  ::inferx::server::scheduler_client::GenerationEvent event;
  event.request_id = "req_contract";
  event.sequence_number = 3;
  event.terminal = true;
  event.finish_reason = ::inferx::server::scheduler_client::FinishReason::kStop;
  event.usage = {.prompt_tokens = 4, .completion_tokens = 2};
  event.embeddings.push_back({.index = 0, .values = {0.25f, -0.5f}});
  EXPECT_EQ(event.usage.prompt_tokens, 4);
  EXPECT_EQ(event.embeddings.front().values.size(), 2);
}

TEST(InProcessSchedulerClientTest, StreamsEventsWithoutBlockingBridge) {
  FakeInProcessBackend backend;
  folly::CPUThreadPoolExecutor coroutine_executor(1);
  ::inferx::server::scheduler_client::InProcessSchedulerClient client(&backend);
  std::thread::id stream_thread;
  auto task = [&]()
      -> folly::coro::Task<
          std::vector<::inferx::server::scheduler_client::GenerationEvent>> {
    stream_thread = std::this_thread::get_id();
    ::inferx::server::scheduler_client::ScheduledRequest command;
    command.request_id = "req_bridge";
    command.model_version = "model@v3";
    auto submitted = co_await client.Submit(command, {});
    EXPECT_TRUE(submitted.ok()) << submitted.status();
    std::vector<::inferx::server::scheduler_client::GenerationEvent> events;
    auto stream = client.Events("req_bridge", {});
    while (auto next = co_await stream.next()) {
      events.push_back(std::move(*next));
    }
    co_return events;
  };
  auto future =
      folly::coro::co_withExecutor(&coroutine_executor, task()).start();
  auto events = std::move(future).get();
  ASSERT_EQ(events.size(), 2);
  EXPECT_EQ(events[0].sequence_number, 1);
  EXPECT_EQ(events[0].text_delta, "delta");
  EXPECT_EQ(events[1].sequence_number, 2);
  EXPECT_TRUE(events[1].terminal);
  EXPECT_EQ(events[1].usage.prompt_tokens, 3);
  EXPECT_EQ(backend.seen_version, "model@v3");
  EXPECT_EQ(backend.generation->next_thread, stream_thread);
}

TEST(InProcessSchedulerClientTest, RejectsEmbeddingWorkloadExplicitly) {
  FakeInProcessBackend backend;
  ::inferx::server::scheduler_client::InProcessSchedulerClient client(&backend);
  ::inferx::server::scheduler_client::ScheduledRequest command;
  command.request_id = "req_embedding";
  command.workload =
      ::inferx::server::scheduler_client::WorkloadClass::kEmbedding;
  auto result = folly::coro::blockingWait(client.Submit(command, {}));
  EXPECT_EQ(result.status().code(), absl::StatusCode::kUnimplemented);
}

TEST(ManagedRequestServiceTest, OwnsLifecycleAdmissionAndFinalization) {
  ::inferx::server::request::RequestManager manager;
  FakeSchedulerClient scheduler;
  ::inferx::server::admission::AdmissionController admission(
      {.max_active_requests = 1,
       .max_reserved_tokens = 16,
       .max_active_per_tenant = 1});
  ::inferx::server::request::ManagedRequestService service(&manager, &scheduler,
                                                           &admission);
  ::inferx::server::scheduler_client::ScheduledRequest command;
  command.request_id = "req_managed";
  command.tenant_id = "tenant";
  command.model_version = "model@v1";
  command.prompt_tokens = {1, 2};
  command.sampling.max_tokens = 3;
  command.deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(10);
  auto submitted = folly::coro::blockingWait(service.Submit(command, {}));
  ASSERT_TRUE(submitted.ok()) << submitted.status();
  EXPECT_EQ(admission.active_requests(), 1);
  EXPECT_EQ(manager.active_count(), 1);
  auto queued = manager.GetStatus("req_managed");
  ASSERT_TRUE(queued.ok()) << queued.status();
  EXPECT_EQ(queued->state, ::inferx::server::request::RequestState::kQueued);

  auto consume = [&]() -> folly::coro::Task<void> {
    auto events = service.Events("req_managed", {});
    while (co_await events.next()) {
    }
  };
  folly::coro::blockingWait(consume());
  EXPECT_EQ(admission.active_requests(), 0);
  EXPECT_EQ(manager.active_count(), 0);
  EXPECT_EQ(manager.completed_count(), 1);
  auto completed = manager.GetStatus("req_managed");
  ASSERT_TRUE(completed.ok()) << completed.status();
  EXPECT_EQ(completed->state,
            ::inferx::server::request::RequestState::kCompleted);
}

TEST(ManagedRequestServiceTest, CancellationIsIdempotentAcrossCleanup) {
  ::inferx::server::request::RequestManager manager;
  FakeSchedulerClient scheduler;
  ::inferx::server::admission::AdmissionController admission({});
  ::inferx::server::request::ManagedRequestService service(&manager, &scheduler,
                                                           &admission);
  ::inferx::server::scheduler_client::ScheduledRequest command;
  command.request_id = "req_cancel";
  command.tenant_id = "tenant";
  command.model_version = "model@v1";
  command.prompt_tokens = {1};
  command.sampling.max_tokens = 1;
  command.deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(10);
  ASSERT_TRUE(folly::coro::blockingWait(service.Submit(command, {})).ok());
  EXPECT_TRUE(
      folly::coro::blockingWait(
          service.Cancel(
              "req_cancel",
              ::inferx::server::request::CancellationReason::kAdministrative))
          .ok());
  EXPECT_TRUE(
      folly::coro::blockingWait(
          service.Cancel(
              "req_cancel",
              ::inferx::server::request::CancellationReason::kAdministrative))
          .ok());
  EXPECT_EQ(scheduler.cancel_count, 1);
  EXPECT_EQ(admission.active_requests(), 0);
  auto snapshot = manager.GetStatus("req_cancel");
  ASSERT_TRUE(snapshot.ok()) << snapshot.status();
  EXPECT_EQ(snapshot->state,
            ::inferx::server::request::RequestState::kCancelled);
}

TEST(HealthHandlerTest, LivenessDoesNotDependOnModelAvailability) {
  using ::inferx::server::handlers::HealthHandler;
  using ::inferx::server::handlers::HealthProbe;
  using ::inferx::server::handlers::HealthState;
  HealthState state;
  state.SetConfigurationLoaded(true);
  state.SetDependenciesLoaded(true);
  state.SetListenerAccepting(true);
  state.SetSchedulerConnected(true);

  HealthHandler live(&state, HealthProbe::kLive);
  CapturingWriter writer;
  HttpRequest request{http::verb::get, "/health/live", 11};
  folly::coro::blockingWait(
      live.Handle(std::move(request), {}, writer, folly::CancellationToken{}));
  EXPECT_EQ(writer.status, 200);
  EXPECT_EQ(writer.body, "{\"status\":\"ok\",\"probe\":\"live\"}");
  EXPECT_FALSE(state.ready());
}

TEST(HealthHandlerTest, ReadinessRequiresEveryServingDependency) {
  using ::inferx::server::handlers::HealthHandler;
  using ::inferx::server::handlers::HealthProbe;
  using ::inferx::server::handlers::HealthState;
  HealthState state;
  state.SetConfigurationLoaded(true);
  state.SetDependenciesLoaded(true);
  state.SetListenerAccepting(true);
  state.SetSchedulerConnected(true);
  HealthHandler ready(&state, HealthProbe::kReady);

  CapturingWriter unavailable;
  HttpRequest first{http::verb::get, "/health/ready", 11};
  folly::coro::blockingWait(ready.Handle(std::move(first), {}, unavailable,
                                         folly::CancellationToken{}));
  EXPECT_EQ(unavailable.status, 503);

  state.SetReadyModelAvailable(true);
  CapturingWriter available;
  HttpRequest second{http::verb::get, "/health/ready", 11};
  folly::coro::blockingWait(ready.Handle(std::move(second), {}, available,
                                         folly::CancellationToken{}));
  EXPECT_EQ(available.status, 200);
}

TEST(ModelsHandlerTest, ListsOnlyReadyModelsVisibleToAuthenticatedTenant) {
  using ::inferx::server::handlers::ModelsHandler;
  using namespace ::inferx::server::model_registry;
  Registry registry;
  ModelRecord visible;
  visible.id = "model-b";
  visible.version = "v1";
  visible.alias = "chat-current";
  visible.created = 42;
  visible.visible_tenants.insert("tenant-a");
  ASSERT_TRUE(registry.Register(visible).ok());
  ASSERT_TRUE(registry.SetState("model-b", "v1", ModelState::kLoading).ok());
  ASSERT_TRUE(registry.SetState("model-b", "v1", ModelState::kWarming).ok());
  ASSERT_TRUE(registry.SetState("model-b", "v1", ModelState::kReady).ok());
  ModelRecord hidden;
  hidden.id = "model-a";
  hidden.version = "v2";
  hidden.visible_tenants.insert("tenant-b");
  ASSERT_TRUE(registry.Register(hidden).ok());
  ASSERT_TRUE(registry.SetState("model-a", "v2", ModelState::kLoading).ok());
  ASSERT_TRUE(registry.SetState("model-a", "v2", ModelState::kWarming).ok());
  ASSERT_TRUE(registry.SetState("model-a", "v2", ModelState::kReady).ok());

  ModelsHandler handler(&registry);
  RequestContext context;
  context.authenticated = true;
  context.tenant_id = "tenant-a";
  CapturingWriter writer;
  HttpRequest request{http::verb::get, "/v1/models", 11};
  folly::coro::blockingWait(
      handler.Handle(std::move(request), std::move(context), writer, {}));
  EXPECT_EQ(writer.status, 200);
  EXPECT_EQ(writer.body,
            "{\"object\":\"list\",\"data\":[{\"id\":\"chat-current\","
            "\"object\":\"model\",\"created\":42,\"owned_by\":\"platform\","
            "\"status\":\"ready\"}]}");
}

TEST(ModelsHandlerTest, RejectsMissingAuthenticatedContext) {
  ::inferx::server::model_registry::Registry registry;
  ::inferx::server::handlers::ModelsHandler handler(&registry);
  CapturingWriter writer;
  HttpRequest request{http::verb::get, "/v1/models", 11};
  folly::coro::blockingWait(handler.Handle(std::move(request), {}, writer, {}));
  EXPECT_EQ(writer.status, 401);
}

TEST(EmbeddingsHandlerTest, PreservesBatchIndicesAndImmutableModelVersion) {
  using namespace ::inferx::server::model_registry;
  Registry registry;
  ModelRecord model;
  model.id = "embed";
  model.version = "v3";
  model.alias = "embed-current";
  model.supports_generation = false;
  model.supports_embeddings = true;
  model.embedding_dimensions = 2;
  model.embedding_max_batch_size = 2;
  model.visible_tenants.insert("tenant-a");
  ASSERT_TRUE(registry.Register(model).ok());
  ASSERT_TRUE(registry.SetState("embed", "v3", ModelState::kLoading).ok());
  ASSERT_TRUE(registry.SetState("embed", "v3", ModelState::kWarming).ok());
  ASSERT_TRUE(registry.SetState("embed", "v3", ModelState::kReady).ok());
  auto service = std::make_shared<FakeEmbeddingsService>();
  ::inferx::server::handlers::EmbeddingsHandler handler(&registry, service);
  RequestContext context;
  context.authenticated = true;
  context.tenant_id = "tenant-a";
  CapturingWriter writer;
  HttpRequest request{http::verb::post, "/v1/embeddings", 11};
  request.body() =
      "{\"model\":\"embed-current\",\"input\":[\"a\",\"b\"],"
      "\"dimensions\":2}";
  folly::coro::blockingWait(
      handler.Handle(std::move(request), std::move(context), writer, {}));
  EXPECT_EQ(writer.status, 200);
  EXPECT_NE(writer.body.find("\"index\":0"), std::string::npos);
  EXPECT_NE(writer.body.find("\"index\":1"), std::string::npos);
  EXPECT_NE(writer.body.find("\"prompt_tokens\":7"), std::string::npos);
  EXPECT_EQ(service->seen_model_version, "v3");
  EXPECT_EQ(service->seen_tenant, "tenant-a");
}

TEST(EmbeddingsHandlerTest, RejectsUnsupportedCapabilityBeforeExecution) {
  using namespace ::inferx::server::model_registry;
  Registry registry;
  ModelRecord model;
  model.id = "chat";
  model.version = "v1";
  model.supports_embeddings = false;
  ASSERT_TRUE(registry.Register(model).ok());
  ASSERT_TRUE(registry.SetState("chat", "v1", ModelState::kLoading).ok());
  ASSERT_TRUE(registry.SetState("chat", "v1", ModelState::kWarming).ok());
  ASSERT_TRUE(registry.SetState("chat", "v1", ModelState::kReady).ok());
  ::inferx::server::handlers::EmbeddingsHandler handler(&registry);
  RequestContext context;
  context.authenticated = true;
  CapturingWriter writer;
  HttpRequest request{http::verb::post, "/v1/embeddings", 11};
  request.body() = "{\"model\":\"chat@v1\",\"input\":\"text\"}";
  folly::coro::blockingWait(
      handler.Handle(std::move(request), std::move(context), writer, {}));
  EXPECT_EQ(writer.status, 422);
  EXPECT_NE(writer.body.find("does not support embedding workloads"),
            std::string::npos);
}

TEST(TokenizeHandlerTest, ResolvesVersionAndDelegatesSpecialTokenPolicy) {
  using namespace ::inferx::server::model_registry;
  Registry registry;
  ModelRecord model;
  model.id = "chat";
  model.version = "v7";
  model.alias = "chat-current";
  model.context_limit = 8;
  model.visible_tenants.insert("tenant-a");
  ASSERT_TRUE(registry.Register(model).ok());
  ASSERT_TRUE(registry.SetState("chat", "v7", ModelState::kLoading).ok());
  ASSERT_TRUE(registry.SetState("chat", "v7", ModelState::kWarming).ok());
  ASSERT_TRUE(registry.SetState("chat", "v7", ModelState::kReady).ok());
  FakeTokenizationService service;
  ::inferx::server::handlers::TokenizeHandler handler(&registry, &service);
  RequestContext context;
  context.authenticated = true;
  context.tenant_id = "tenant-a";
  CapturingWriter writer;
  HttpRequest request{http::verb::post, "/v1/tokenize", 11};
  request.body() =
      "{\"model\":\"chat-current\",\"text\":\"hello\","
      "\"add_special_tokens\":true}";
  folly::coro::blockingWait(
      handler.Handle(std::move(request), std::move(context), writer, {}));
  EXPECT_EQ(writer.status, 200);
  EXPECT_EQ(writer.body,
            "{\"model\":\"chat-current\",\"token_ids\":[11,12,13],"
            "\"token_count\":3}");
  EXPECT_EQ(service.seen_version, "chat@v7");
  EXPECT_EQ(service.seen_prompt, "hello");
  EXPECT_TRUE(service.seen_add_special_tokens);
}

TEST(TokenizeHandlerTest, EnforcesExactTokenContextLimit) {
  using namespace ::inferx::server::model_registry;
  Registry registry;
  ModelRecord model;
  model.id = "small";
  model.version = "v1";
  model.context_limit = 2;
  ASSERT_TRUE(registry.Register(model).ok());
  ASSERT_TRUE(registry.SetState("small", "v1", ModelState::kLoading).ok());
  ASSERT_TRUE(registry.SetState("small", "v1", ModelState::kWarming).ok());
  ASSERT_TRUE(registry.SetState("small", "v1", ModelState::kReady).ok());
  FakeTokenizationService service;
  ::inferx::server::handlers::TokenizeHandler handler(&registry, &service);
  RequestContext context;
  context.authenticated = true;
  CapturingWriter writer;
  HttpRequest request{http::verb::post, "/v1/tokenize", 11};
  request.body() = "{\"model\":\"small@v1\",\"text\":\"hello\"}";
  folly::coro::blockingWait(
      handler.Handle(std::move(request), std::move(context), writer, {}));
  EXPECT_EQ(writer.status, 400);
  EXPECT_NE(writer.body.find("context limit"), std::string::npos);
}

TEST(ModelRetrieveHandlerTest, ReturnsSingleModelOrOpenAiNotFound) {
  using namespace ::inferx::server::model_registry;
  Registry registry;
  ModelRecord model;
  model.id = "chat";
  model.version = "v7";
  model.alias = "chat-current";
  model.created = 1712;
  model.visible_tenants.insert("tenant-a");
  ASSERT_TRUE(registry.Register(model).ok());
  ASSERT_TRUE(registry.SetState("chat", "v7", ModelState::kLoading).ok());
  ASSERT_TRUE(registry.SetState("chat", "v7", ModelState::kWarming).ok());
  ASSERT_TRUE(registry.SetState("chat", "v7", ModelState::kReady).ok());
  ::inferx::server::handlers::ModelRetrieveHandler handler(&registry,
                                                           "/v1/models/");
  RequestContext context;
  context.authenticated = true;
  context.tenant_id = "tenant-a";

  CapturingWriter writer;
  HttpRequest request{http::verb::get, "/v1/models/chat-current", 11};
  folly::coro::blockingWait(
      handler.Handle(std::move(request), context, writer, {}));
  EXPECT_EQ(writer.status, 200);
  EXPECT_EQ(writer.body,
            "{\"id\":\"chat-current\",\"object\":\"model\",\"created\":1712,"
            "\"owned_by\":\"inferx\"}");

  CapturingWriter missing_writer;
  HttpRequest missing{http::verb::get, "/v1/models/unknown", 11};
  folly::coro::blockingWait(
      handler.Handle(std::move(missing), context, missing_writer, {}));
  EXPECT_EQ(missing_writer.status, 404);
  EXPECT_NE(missing_writer.body.find("\"code\":\"model_not_found\""),
            std::string::npos);
  EXPECT_NE(missing_writer.body.find("\"param\":\"model\""),
            std::string::npos);
}

TEST(VllmTokenizeHandlerTest, EmitsVllmShapeAndRejectsChatForm) {
  using namespace ::inferx::server::model_registry;
  Registry registry;
  ModelRecord model;
  model.id = "chat";
  model.version = "v7";
  model.alias = "chat-current";
  model.visible_tenants.insert("tenant-a");
  ASSERT_TRUE(registry.Register(model).ok());
  ASSERT_TRUE(registry.SetState("chat", "v7", ModelState::kLoading).ok());
  ASSERT_TRUE(registry.SetState("chat", "v7", ModelState::kWarming).ok());
  ASSERT_TRUE(registry.SetState("chat", "v7", ModelState::kReady).ok());
  FakeTokenizationService service;
  ::inferx::server::handlers::VllmTokenizeHandler handler(&registry, &service);
  RequestContext context;
  context.authenticated = true;
  context.tenant_id = "tenant-a";

  CapturingWriter writer;
  HttpRequest request{http::verb::post, "/tokenize", 11};
  request.body() = "{\"model\":\"chat-current\",\"prompt\":\"hello\"}";
  folly::coro::blockingWait(
      handler.Handle(std::move(request), context, writer, {}));
  EXPECT_EQ(writer.status, 200);
  EXPECT_EQ(writer.body,
            "{\"count\":3,\"max_model_len\":null,\"tokens\":[11,12,13]}");
  EXPECT_EQ(service.seen_version, "chat@v7");
  EXPECT_EQ(service.seen_prompt, "hello");
  // vLLM's default, the opposite of /v1/tokenize's.
  EXPECT_TRUE(service.seen_add_special_tokens);

  CapturingWriter chat_writer;
  HttpRequest chat{http::verb::post, "/tokenize", 11};
  chat.body() =
      "{\"model\":\"chat-current\",\"messages\":[{\"role\":\"user\","
      "\"content\":\"hi\"}]}";
  folly::coro::blockingWait(
      handler.Handle(std::move(chat), context, chat_writer, {}));
  EXPECT_EQ(chat_writer.status, 400);
  EXPECT_NE(chat_writer.body.find(
                "chat tokenization via /tokenize is not supported"),
            std::string::npos);
}

TEST(DetokenizeHandlerTest, RoundTripsTokenizeOutputThroughSequenceDecode) {
  using namespace ::inferx::server::model_registry;
  Registry registry;
  ModelRecord model;
  model.id = "chat";
  model.version = "v7";
  model.alias = "chat-current";
  model.visible_tenants.insert("tenant-a");
  ASSERT_TRUE(registry.Register(model).ok());
  ASSERT_TRUE(registry.SetState("chat", "v7", ModelState::kLoading).ok());
  ASSERT_TRUE(registry.SetState("chat", "v7", ModelState::kWarming).ok());
  ASSERT_TRUE(registry.SetState("chat", "v7", ModelState::kReady).ok());
  FakeTokenizationService service;
  ::inferx::server::handlers::VllmTokenizeHandler tokenize(&registry,
                                                           &service);
  ::inferx::server::handlers::DetokenizeHandler detokenize(&registry,
                                                           &service);
  RequestContext context;
  context.authenticated = true;
  context.tenant_id = "tenant-a";

  CapturingWriter tokenize_writer;
  HttpRequest encode{http::verb::post, "/tokenize", 11};
  encode.body() = "{\"model\":\"chat-current\",\"prompt\":\"hello\"}";
  folly::coro::blockingWait(
      tokenize.Handle(std::move(encode), context, tokenize_writer, {}));
  ASSERT_EQ(tokenize_writer.status, 200);

  CapturingWriter detokenize_writer;
  HttpRequest decode{http::verb::post, "/detokenize", 11};
  decode.body() = "{\"model\":\"chat-current\",\"tokens\":[11,12,13]}";
  folly::coro::blockingWait(
      detokenize.Handle(std::move(decode), context, detokenize_writer, {}));
  EXPECT_EQ(detokenize_writer.status, 200);
  EXPECT_EQ(detokenize_writer.body, "{\"prompt\":\"hello\"}");
  EXPECT_EQ(service.seen_version, "chat@v7");
  EXPECT_EQ(service.seen_detokenize_ids,
            (std::vector<::inferx::tokenizer::TokenId>{11, 12, 13}));

  CapturingWriter malformed_writer;
  HttpRequest malformed{http::verb::post, "/detokenize", 11};
  malformed.body() = "{\"model\":\"chat-current\",\"tokens\":\"11\"}";
  folly::coro::blockingWait(
      detokenize.Handle(std::move(malformed), context, malformed_writer, {}));
  EXPECT_EQ(malformed_writer.status, 400);
  EXPECT_NE(malformed_writer.body.find("array of integers"),
            std::string::npos);
}

TEST(ApiRoutesTest, ComposesPublicProbesAndScopedTenantApi) {
  auto store = std::make_shared<::inferx::server::auth::ApiKeyStore>();
  ::inferx::server::auth::Principal principal{"tenant-a", "user", "key"};
  principal.scopes.insert("models.read");
  principal.scopes.insert("inference.invoke");
  ASSERT_TRUE(store
                  ->AddHash("2bb80d537b1da3e38bd30361aa855686bde0eacd7162fef6a2"
                            "5fe97bf527a25b",
                            principal)
                  .ok());
  auto authenticator =
      std::make_shared<::inferx::server::auth::ApiKeyAuthenticator>(
          store.get());
  auto guard = std::make_shared<::inferx::server::middleware::BearerRouteGuard>(
      authenticator);
  ::inferx::server::handlers::HealthState health;
  ::inferx::server::model_registry::Registry registry;
  FakeTokenizationService tokenization;
  FakeRequestService requests;
  FakeMetricsSource metrics;
  auto built = ::inferx::server::handlers::BuildApiRoutes(
      {.health = &health,
       .models = &registry,
       .tokenization = &tokenization,
       .requests = &requests,
       .metrics = &metrics,
       .guard = guard,
       .max_inference_body_bytes = 64});
  ASSERT_TRUE(built.ok()) << built.status();

  CapturingWriter live_writer;
  HttpRequest live{http::verb::get, "/health/live", 11};
  folly::coro::blockingWait(
      (*built)->Handle(std::move(live), {}, live_writer, {}));
  EXPECT_EQ(live_writer.status, 200);

  CapturingWriter compatibility_writer;
  HttpRequest compatibility{http::verb::get, "/health", 11};
  folly::coro::blockingWait(
      (*built)->Handle(std::move(compatibility), {}, compatibility_writer, {}));
  EXPECT_EQ(compatibility_writer.status, 200);

  CapturingWriter denied_writer;
  HttpRequest denied{http::verb::get, "/v1/models", 11};
  folly::coro::blockingWait(
      (*built)->Handle(std::move(denied), {}, denied_writer, {}));
  EXPECT_EQ(denied_writer.status, 401);

  CapturingWriter models_writer;
  HttpRequest models{http::verb::get, "/v1/models", 11};
  models.set(http::field::authorization, "Bearer secret");
  folly::coro::blockingWait(
      (*built)->Handle(std::move(models), {}, models_writer, {}));
  EXPECT_EQ(models_writer.status, 200);
  EXPECT_EQ(models_writer.body, "{\"object\":\"list\",\"data\":[]}");

  // The retrieve prefix route shares the listing's scope; the empty registry
  // answers with the OpenAI-shaped 404 rather than the router's.
  CapturingWriter retrieve_writer;
  HttpRequest retrieve{http::verb::get, "/v1/models/unknown", 11};
  retrieve.set(http::field::authorization, "Bearer secret");
  folly::coro::blockingWait(
      (*built)->Handle(std::move(retrieve), {}, retrieve_writer, {}));
  EXPECT_EQ(retrieve_writer.status, 404);
  EXPECT_NE(retrieve_writer.body.find("\"code\":\"model_not_found\""),
            std::string::npos);
}

TEST(ApiRoutesTest, ServesVllmCompatibilitySurface) {
  auto guard = std::make_shared<ScopeGuard>();
  ::inferx::server::handlers::HealthState health;
  ::inferx::server::model_registry::Registry registry;
  FakeTokenizationService tokenization;
  FakeRequestService requests;
  FakeMetricsSource metrics;
  auto built = ::inferx::server::handlers::BuildApiRoutes(
      {.health = &health,
       .models = &registry,
       .tokenization = &tokenization,
       .requests = &requests,
       .metrics = &metrics,
       .guard = guard,
       .max_inference_body_bytes = 64});
  ASSERT_TRUE(built.ok()) << built.status();

  CapturingWriter version_writer;
  HttpRequest version{http::verb::get, "/version", 11};
  folly::coro::blockingWait(
      (*built)->Handle(std::move(version), {}, version_writer, {}));
  EXPECT_EQ(version_writer.status, 200);
  // Shape only: pinning the number would make every version bump a test edit.
  EXPECT_TRUE(version_writer.body.starts_with("{\"version\":\""));
  EXPECT_TRUE(version_writer.body.ends_with("\"}"));
  EXPECT_GT(version_writer.body.size(), std::string("{\"version\":\"\"}").size());

  for (const auto method : {http::verb::get, http::verb::post}) {
    CapturingWriter ping_writer;
    HttpRequest ping{method, "/ping", 11};
    folly::coro::blockingWait(
        (*built)->Handle(std::move(ping), {}, ping_writer, {}));
    EXPECT_EQ(ping_writer.status, 200);
    EXPECT_EQ(ping_writer.body, "{}");
  }

  CapturingWriter score_writer;
  HttpRequest score{http::verb::post, "/v1/score", 11};
  score.body() = "{\"model\":\"m\",\"text_1\":\"a\",\"text_2\":\"b\"}";
  folly::coro::blockingWait(
      (*built)->Handle(std::move(score), {}, score_writer, {}));
  EXPECT_EQ(score_writer.status, 501);
  EXPECT_NE(score_writer.body.find("\"type\":\"not_supported_error\""),
            std::string::npos);
  EXPECT_NE(score_writer.body.find(
                "/v1/score is not supported by InferX (pooling models)"),
            std::string::npos);
}

TEST(MetricsHandlerTest, RendersPrometheusAndLegacyStatsFromSameSnapshot) {
  FakeMetricsSource source;
  ::inferx::server::handlers::MetricsHandler metrics(
      &source, ::inferx::server::handlers::MetricsPresentation::kPrometheus);
  CapturingWriter metrics_writer;
  HttpRequest metrics_request{http::verb::get, "/metrics", 11};
  folly::coro::blockingWait(
      metrics.Handle(std::move(metrics_request), {}, metrics_writer, {}));
  EXPECT_EQ(metrics_writer.status, 200);
  EXPECT_NE(metrics_writer.body.find("inferx_requests_running 2"),
            std::string::npos);
  EXPECT_NE(metrics_writer.body.find("inferx_kv_blocks{state=\"free\"} 5"),
            std::string::npos);
  EXPECT_EQ(metrics_writer.body.find("tenant"), std::string::npos);

  ::inferx::server::handlers::MetricsHandler stats(
      &source, ::inferx::server::handlers::MetricsPresentation::kLegacyJson);
  CapturingWriter stats_writer;
  HttpRequest stats_request{http::verb::get, "/stats", 11};
  folly::coro::blockingWait(
      stats.Handle(std::move(stats_request), {}, stats_writer, {}));
  EXPECT_EQ(stats_writer.status, 200);
  EXPECT_NE(stats_writer.body.find("\"running\":2"), std::string::npos);
  EXPECT_NE(stats_writer.body.find("\"tokens_generated\":10"),
            std::string::npos);
}

TEST(AdminModelHandlerTest, UsesDistinctScopeAndListsTenantLifecycle) {
  auto store = std::make_shared<::inferx::server::auth::ApiKeyStore>();
  ::inferx::server::auth::Principal principal{"tenant-a", "operator", "key"};
  principal.scopes.insert("models.manage");
  ASSERT_TRUE(store
                  ->AddHash("2bb80d537b1da3e38bd30361aa855686bde0eacd7162fef6a2"
                            "5fe97bf527a25b",
                            principal)
                  .ok());
  auto authenticator =
      std::make_shared<::inferx::server::auth::ApiKeyAuthenticator>(
          store.get());
  auto guard = std::make_shared<::inferx::server::middleware::BearerRouteGuard>(
      authenticator);
  using namespace ::inferx::server::model_registry;
  Registry registry;
  ModelRecord visible;
  visible.id = "model";
  visible.version = "v2";
  visible.alias = "current";
  visible.visible_tenants.insert("tenant-a");
  ASSERT_TRUE(registry.Register(visible).ok());
  ASSERT_TRUE(registry.SetState("model", "v2", ModelState::kLoading).ok());
  ModelRecord hidden;
  hidden.id = "hidden";
  hidden.version = "v1";
  hidden.visible_tenants.insert("tenant-b");
  ASSERT_TRUE(registry.Register(hidden).ok());
  auto routes = ::inferx::server::handlers::BuildAdminRoutes(&registry, guard);
  ASSERT_TRUE(routes.ok()) << routes.status();

  CapturingWriter list_writer;
  HttpRequest list{http::verb::get, "/admin/v1/models", 11};
  list.set(http::field::authorization, "Bearer secret");
  folly::coro::blockingWait(
      (*routes)->Handle(std::move(list), {}, list_writer, {}));
  EXPECT_EQ(list_writer.status, 200);
  EXPECT_NE(list_writer.body.find("\"state\":\"loading\""), std::string::npos);
  EXPECT_EQ(list_writer.body.find("hidden"), std::string::npos);

  CapturingWriter load_writer;
  HttpRequest load{http::verb::post, "/admin/v1/models/load", 11};
  load.set(http::field::authorization, "Bearer secret");
  load.body() = "{}";
  folly::coro::blockingWait(
      (*routes)->Handle(std::move(load), {}, load_writer, {}));
  EXPECT_EQ(load_writer.status, 422);
  EXPECT_NE(load_writer.body.find("not implemented"), std::string::npos);
}

TEST(ServerConfigTest, RejectsUnsafeListenerAndAdmissionCombinations) {
  using ::inferx::server::config::ServerConfig;
  using ::inferx::server::config::Validate;
  ServerConfig config;
  config.development_auth_bypass = true;
  EXPECT_FALSE(Validate(config).ok());
  config.listen_address = "127.0.0.1";
  EXPECT_TRUE(Validate(config).ok());
  config.admin_enabled = true;
  config.admin_listen_address = "127.0.0.1";
  config.admin_port = config.port;
  EXPECT_FALSE(Validate(config).ok());
  config.admin_port = 9000;
  config.max_active_per_tenant = config.max_active_requests + 1;
  EXPECT_FALSE(Validate(config).ok());
}

TEST(ServerConfigTest, ReloadsPolicyAtomicallyAndRejectsTopologyChanges) {
  using ::inferx::server::config::ConfigStore;
  using ::inferx::server::config::ServerConfig;
  ServerConfig initial;
  auto created = ConfigStore::Create(initial);
  ASSERT_TRUE(created.ok()) << created.status();
  auto store = *created;
  ServerConfig policy = initial;
  policy.max_active_requests = 512;
  policy.max_active_per_tenant = 64;
  policy.completed_retention = std::chrono::seconds(60);
  ASSERT_TRUE(store->Reload(policy).ok());
  EXPECT_EQ(store->revision(), 2);
  const auto snapshot = store->Snapshot();
  EXPECT_EQ(snapshot->max_active_requests, 512);
  EXPECT_EQ(snapshot->completed_retention, std::chrono::seconds(60));

  ServerConfig topology = policy;
  topology.port = 9000;
  EXPECT_EQ(store->Reload(topology).code(),
            absl::StatusCode::kFailedPrecondition);
  EXPECT_EQ(store->revision(), 2);
  EXPECT_EQ(store->Snapshot()->port, initial.port);
}

TEST(CompletionHandlerTest, CollectsTypedEventsFromImmutableModelRequest) {
  using namespace ::inferx::server::model_registry;
  Registry registry;
  ModelRecord model;
  model.id = "text";
  model.version = "v4";
  model.alias = "text-current";
  model.context_limit = 8;
  model.tokenizer_revision = "tok-v2";
  model.visible_tenants.insert("tenant-a");
  ASSERT_TRUE(registry.Register(model).ok());
  ASSERT_TRUE(registry.SetState("text", "v4", ModelState::kLoading).ok());
  ASSERT_TRUE(registry.SetState("text", "v4", ModelState::kWarming).ok());
  ASSERT_TRUE(registry.SetState("text", "v4", ModelState::kReady).ok());
  FakeTokenizationService tokenization;
  FakeRequestService requests;
  ::inferx::server::handlers::CompletionHandler handler(
      &registry, &tokenization, &requests);
  RequestContext context;
  context.authenticated = true;
  context.tenant_id = "tenant-a";
  CapturingWriter writer;
  HttpRequest request{http::verb::post, "/v1/completions", 11};
  request.body() =
      "{\"model\":\"text-current\",\"prompt\":\"input\","
      "\"max_tokens\":2}";
  folly::coro::blockingWait(
      handler.Handle(std::move(request), std::move(context), writer, {}));
  EXPECT_EQ(writer.status, 200);
  EXPECT_NE(writer.body.find("\"text\":\"hello\""), std::string::npos);
  EXPECT_NE(writer.body.find("\"prompt_tokens\":3"), std::string::npos);
  EXPECT_EQ(requests.submitted.model_version, "text@v4");
  EXPECT_EQ(requests.submitted.tokenizer_revision, "tok-v2");
  EXPECT_EQ(requests.submitted.tenant_id, "tenant-a");
  EXPECT_EQ(requests.submitted.prompt_tokens.size(), 3);
}

TEST(CompletionHandlerTest, StreamsSameEventsWithUsageAndDoneMarker) {
  using namespace ::inferx::server::model_registry;
  Registry registry;
  ModelRecord model;
  model.id = "text";
  model.version = "v1";
  model.context_limit = 8;
  ASSERT_TRUE(registry.Register(model).ok());
  ASSERT_TRUE(registry.SetState("text", "v1", ModelState::kLoading).ok());
  ASSERT_TRUE(registry.SetState("text", "v1", ModelState::kWarming).ok());
  ASSERT_TRUE(registry.SetState("text", "v1", ModelState::kReady).ok());
  FakeTokenizationService tokenization;
  FakeRequestService requests;
  ::inferx::server::handlers::CompletionHandler handler(
      &registry, &tokenization, &requests);
  RequestContext context;
  context.authenticated = true;
  CapturingWriter writer;
  HttpRequest request{http::verb::post, "/v1/completions", 11};
  request.body() =
      "{\"model\":\"text@v1\",\"prompt\":\"input\","
      "\"max_tokens\":2,\"stream\":true,"
      "\"stream_options\":{\"include_usage\":true}}";
  folly::coro::blockingWait(
      handler.Handle(std::move(request), std::move(context), writer, {}));
  EXPECT_EQ(writer.status, 200);
  EXPECT_TRUE(writer.finished);
  EXPECT_NE(writer.body.find("hello"), std::string::npos);
  EXPECT_NE(writer.body.find("\"choices\":[]"), std::string::npos);
  EXPECT_NE(writer.body.find("data: [DONE]"), std::string::npos);
}

TEST(ChatCompletionHandlerTest, UsesChatTokenizerAndCollectsAssistantMessage) {
  using namespace ::inferx::server::model_registry;
  Registry registry;
  ModelRecord model;
  model.id = "chat";
  model.version = "v2";
  model.alias = "chat-current";
  model.context_limit = 8;
  model.tokenizer_revision = "chat-tok";
  ASSERT_TRUE(registry.Register(model).ok());
  ASSERT_TRUE(registry.SetState("chat", "v2", ModelState::kLoading).ok());
  ASSERT_TRUE(registry.SetState("chat", "v2", ModelState::kWarming).ok());
  ASSERT_TRUE(registry.SetState("chat", "v2", ModelState::kReady).ok());
  FakeTokenizationService tokenization;
  FakeRequestService requests;
  ::inferx::server::handlers::ChatCompletionHandler handler(
      &registry, &tokenization, &requests);
  RequestContext context;
  context.authenticated = true;
  CapturingWriter writer;
  HttpRequest request{http::verb::post, "/v1/chat/completions", 11};
  request.body() =
      "{\"model\":\"chat-current\",\"messages\":[{\"role\":\"user\","
      "\"content\":\"hi\"}],\"max_tokens\":2}";
  folly::coro::blockingWait(
      handler.Handle(std::move(request), std::move(context), writer, {}));
  EXPECT_EQ(writer.status, 200);
  EXPECT_NE(writer.body.find("\"role\":\"assistant\""), std::string::npos);
  EXPECT_NE(writer.body.find("\"content\":\"hello\""), std::string::npos);
  EXPECT_EQ(tokenization.seen_version, "chat@v2");
  EXPECT_EQ(tokenization.seen_messages, 1);
  EXPECT_EQ(requests.submitted.tokenizer_revision, "chat-tok");
}

TEST(ChatCompletionHandlerTest, StreamsRoleBeforeContentAndUsage) {
  using namespace ::inferx::server::model_registry;
  Registry registry;
  ModelRecord model;
  model.id = "chat";
  model.version = "v1";
  model.context_limit = 8;
  ASSERT_TRUE(registry.Register(model).ok());
  ASSERT_TRUE(registry.SetState("chat", "v1", ModelState::kLoading).ok());
  ASSERT_TRUE(registry.SetState("chat", "v1", ModelState::kWarming).ok());
  ASSERT_TRUE(registry.SetState("chat", "v1", ModelState::kReady).ok());
  FakeTokenizationService tokenization;
  FakeRequestService requests;
  ::inferx::server::handlers::ChatCompletionHandler handler(
      &registry, &tokenization, &requests);
  RequestContext context;
  context.authenticated = true;
  CapturingWriter writer;
  HttpRequest request{http::verb::post, "/v1/chat/completions", 11};
  request.body() =
      "{\"model\":\"chat@v1\",\"messages\":[{\"role\":\"user\","
      "\"content\":\"hi\"}],\"max_tokens\":2,\"stream\":true,"
      "\"stream_options\":{\"include_usage\":true}}";
  folly::coro::blockingWait(
      handler.Handle(std::move(request), std::move(context), writer, {}));
  EXPECT_EQ(writer.status, 200);
  const size_t role = writer.body.find("\"role\":\"assistant\"");
  const size_t content = writer.body.find("hello");
  ASSERT_NE(role, std::string::npos);
  ASSERT_NE(content, std::string::npos);
  EXPECT_LT(role, content);
  EXPECT_NE(writer.body.find("\"choices\":[]"), std::string::npos);
  EXPECT_NE(writer.body.find("data: [DONE]"), std::string::npos);
  EXPECT_TRUE(writer.finished);
}

// Registers a ready generation model "text@v1"; Registry owns a mutex, so it
// is populated in place rather than returned.
void FillReadyTextRegistry(::inferx::server::model_registry::Registry* registry) {
  using namespace ::inferx::server::model_registry;
  ModelRecord model;
  model.id = "text";
  model.version = "v1";
  model.context_limit = 64;
  ASSERT_TRUE(registry->Register(model).ok());
  ASSERT_TRUE(registry->SetState("text", "v1", ModelState::kLoading).ok());
  ASSERT_TRUE(registry->SetState("text", "v1", ModelState::kWarming).ok());
  ASSERT_TRUE(registry->SetState("text", "v1", ModelState::kReady).ok());
}

TEST(CompletionHandlerTest, RejectsStreamingWithMultipleChoices) {
  ::inferx::server::model_registry::Registry registry;
  FillReadyTextRegistry(&registry);
  FakeTokenizationService tokenization;
  FakeRequestService requests;
  ::inferx::server::handlers::CompletionHandler handler(
      &registry, &tokenization, &requests);
  RequestContext context;
  context.authenticated = true;
  CapturingWriter writer;
  HttpRequest request{http::verb::post, "/v1/completions", 11};
  request.body() =
      "{\"model\":\"text@v1\",\"prompt\":\"input\",\"n\":2,\"stream\":true}";
  folly::coro::blockingWait(
      handler.Handle(std::move(request), std::move(context), writer, {}));
  EXPECT_EQ(writer.status, 400);
  EXPECT_NE(writer.body.find("streaming with n>1"), std::string::npos);
  EXPECT_TRUE(requests.all_submitted.empty());
}

TEST(CompletionHandlerTest, RejectsChoiceCountAboveServerCap) {
  ::inferx::server::model_registry::Registry registry;
  FillReadyTextRegistry(&registry);
  FakeTokenizationService tokenization;
  FakeRequestService requests;
  ::inferx::server::handlers::CompletionHandler handler(
      &registry, &tokenization, &requests);
  RequestContext context;
  context.authenticated = true;
  CapturingWriter writer;
  HttpRequest request{http::verb::post, "/v1/completions", 11};
  request.body() = "{\"model\":\"text@v1\",\"prompt\":\"input\",\"n\":9}";
  folly::coro::blockingWait(
      handler.Handle(std::move(request), std::move(context), writer, {}));
  EXPECT_EQ(writer.status, 400);
  EXPECT_NE(writer.body.find("parallel-choice limit"), std::string::npos);
  EXPECT_TRUE(requests.all_submitted.empty());
}

TEST(CompletionHandlerTest, RunsParallelChoicesWithDerivedSeeds) {
  ::inferx::server::model_registry::Registry registry;
  FillReadyTextRegistry(&registry);
  FakeTokenizationService tokenization;
  FakeRequestService requests;
  ::inferx::server::handlers::CompletionHandler handler(
      &registry, &tokenization, &requests);
  RequestContext context;
  context.authenticated = true;
  CapturingWriter writer;
  HttpRequest request{http::verb::post, "/v1/completions", 11};
  request.body() =
      "{\"model\":\"text@v1\",\"prompt\":\"input\",\"max_tokens\":2,"
      "\"n\":2,\"seed\":9}";
  folly::coro::blockingWait(
      handler.Handle(std::move(request), std::move(context), writer, {}));
  EXPECT_EQ(writer.status, 200);
  EXPECT_NE(writer.body.find("\"index\":0"), std::string::npos);
  EXPECT_NE(writer.body.find("\"index\":1"), std::string::npos);
  // Prompt tokens count once; each choice's single completion token sums.
  EXPECT_NE(writer.body.find("\"usage\":{\"prompt_tokens\":3,"
                             "\"completion_tokens\":2,\"total_tokens\":5}"),
            std::string::npos);
  ASSERT_EQ(requests.all_submitted.size(), 2);
  EXPECT_EQ(requests.all_submitted[0].sampling.seed, 9u);
  EXPECT_EQ(requests.all_submitted[1].sampling.seed, 10u);
  EXPECT_NE(requests.all_submitted[0].request_id,
            requests.all_submitted[1].request_id);
}

TEST(CompletionHandlerTest, EchoPrependsPromptText) {
  ::inferx::server::model_registry::Registry registry;
  FillReadyTextRegistry(&registry);
  FakeTokenizationService tokenization;
  FakeRequestService requests;
  ::inferx::server::handlers::CompletionHandler handler(
      &registry, &tokenization, &requests);
  RequestContext context;
  context.authenticated = true;
  CapturingWriter writer;
  HttpRequest request{http::verb::post, "/v1/completions", 11};
  request.body() =
      "{\"model\":\"text@v1\",\"prompt\":\"input\",\"max_tokens\":2,"
      "\"echo\":true}";
  folly::coro::blockingWait(
      handler.Handle(std::move(request), std::move(context), writer, {}));
  EXPECT_EQ(writer.status, 200);
  EXPECT_NE(writer.body.find("\"text\":\"inputhello\""), std::string::npos);
}

TEST(CompletionHandlerTest, StreamingEchoLeadsWithPromptChunk) {
  ::inferx::server::model_registry::Registry registry;
  FillReadyTextRegistry(&registry);
  FakeTokenizationService tokenization;
  FakeRequestService requests;
  ::inferx::server::handlers::CompletionHandler handler(
      &registry, &tokenization, &requests);
  RequestContext context;
  context.authenticated = true;
  CapturingWriter writer;
  HttpRequest request{http::verb::post, "/v1/completions", 11};
  request.body() =
      "{\"model\":\"text@v1\",\"prompt\":\"input\",\"max_tokens\":2,"
      "\"echo\":true,\"stream\":true}";
  folly::coro::blockingWait(
      handler.Handle(std::move(request), std::move(context), writer, {}));
  EXPECT_EQ(writer.status, 200);
  const size_t prompt = writer.body.find("\"text\":\"input\"");
  const size_t generated = writer.body.find("\"text\":\"hello\"");
  ASSERT_NE(prompt, std::string::npos);
  ASSERT_NE(generated, std::string::npos);
  EXPECT_LT(prompt, generated);
}

TEST(CompletionHandlerTest, ReportsLogprobsWithFallbackTokenSpelling) {
  ::inferx::server::model_registry::Registry registry;
  FillReadyTextRegistry(&registry);
  FakeTokenizationService tokenization;
  FakeRequestService requests;
  requests.emit_logprobs = true;
  ::inferx::server::handlers::CompletionHandler handler(
      &registry, &tokenization, &requests);
  RequestContext context;
  context.authenticated = true;
  CapturingWriter writer;
  HttpRequest request{http::verb::post, "/v1/completions", 11};
  request.body() =
      "{\"model\":\"text@v1\",\"prompt\":\"input\",\"max_tokens\":2,"
      "\"logprobs\":2}";
  folly::coro::blockingWait(
      handler.Handle(std::move(request), std::move(context), writer, {}));
  EXPECT_EQ(writer.status, 200);
  // The fake service has no tokenizer, so token text falls back to the
  // placeholder spelling; both generated positions must be reported.
  EXPECT_NE(writer.body.find("\"tokens\":[\"token_id:5\",\"token_id:7\"]"),
            std::string::npos);
  EXPECT_NE(writer.body.find("\"token_logprobs\":[-0.250000,-0.500000]"),
            std::string::npos);
  EXPECT_NE(
      writer.body.find(
          "\"top_logprobs\":[{\"token_id:5\":-0.250000,"
          "\"token_id:6\":-1.500000},{}]"),
      std::string::npos);
  EXPECT_TRUE(requests.submitted.sampling.want_logprobs);
  EXPECT_EQ(requests.submitted.sampling.top_logprobs, 2);
}

TEST(CompletionHandlerTest, StreamsChunkForTokenWithoutText) {
  ::inferx::server::model_registry::Registry registry;
  FillReadyTextRegistry(&registry);
  FakeTokenizationService tokenization;
  FakeRequestService requests;
  requests.emit_logprobs = true;
  ::inferx::server::handlers::CompletionHandler handler(
      &registry, &tokenization, &requests);
  RequestContext context;
  context.authenticated = true;
  CapturingWriter writer;
  HttpRequest request{http::verb::post, "/v1/completions", 11};
  request.body() =
      "{\"model\":\"text@v1\",\"prompt\":\"input\",\"max_tokens\":2,"
      "\"logprobs\":0,\"stream\":true}";
  folly::coro::blockingWait(
      handler.Handle(std::move(request), std::move(context), writer, {}));
  EXPECT_EQ(writer.status, 200);
  // The second token completed no character but its report still needs a
  // chunk: an empty text with a non-null logprobs object.
  EXPECT_NE(writer.body.find("\"text\":\"\",\"logprobs\":{\"tokens\":"
                             "[\"token_id:7\"]"),
            std::string::npos);
  EXPECT_NE(writer.body.find("\"tokens\":[\"token_id:5\"]"),
            std::string::npos);
}

TEST(ChatCompletionHandlerTest, ReportsLogprobsInOpenAiChatShape) {
  using namespace ::inferx::server::model_registry;
  Registry registry;
  ModelRecord model;
  model.id = "chat";
  model.version = "v1";
  model.context_limit = 64;
  ASSERT_TRUE(registry.Register(model).ok());
  ASSERT_TRUE(registry.SetState("chat", "v1", ModelState::kLoading).ok());
  ASSERT_TRUE(registry.SetState("chat", "v1", ModelState::kWarming).ok());
  ASSERT_TRUE(registry.SetState("chat", "v1", ModelState::kReady).ok());
  FakeTokenizationService tokenization;
  FakeRequestService requests;
  requests.emit_logprobs = true;
  ::inferx::server::handlers::ChatCompletionHandler handler(
      &registry, &tokenization, &requests);
  RequestContext context;
  context.authenticated = true;
  CapturingWriter writer;
  HttpRequest request{http::verb::post, "/v1/chat/completions", 11};
  request.body() =
      "{\"model\":\"chat@v1\",\"messages\":[{\"role\":\"user\","
      "\"content\":\"hi\"}],\"max_tokens\":2,\"logprobs\":true,"
      "\"top_logprobs\":2}";
  folly::coro::blockingWait(
      handler.Handle(std::move(request), std::move(context), writer, {}));
  EXPECT_EQ(writer.status, 200);
  EXPECT_NE(
      writer.body.find(
          "\"logprobs\":{\"content\":[{\"token\":\"token_id:5\","
          "\"logprob\":-0.250000"),
      std::string::npos);
  EXPECT_NE(writer.body.find("{\"token\":\"token_id:6\","
                             "\"logprob\":-1.500000"),
            std::string::npos);
}

TEST(BeastListenerTest, ServesSequentialKeepAliveRequests) {
  auto runtime_result = IoRuntime::Create({1, 1, 1});
  ASSERT_TRUE(runtime_result.ok()) << runtime_result.status();
  auto runtime = std::move(*runtime_result);
  BeastListenerConfig config;
  config.max_request_bytes = 4;
  auto listener_result = BeastListener::Create(runtime.get(), config,
                                               std::make_shared<OkHandler>());
  ASSERT_TRUE(listener_result.ok()) << listener_result.status();
  auto listener = std::move(*listener_result);
  const Status listener_status = listener->Start();
  ASSERT_TRUE(listener_status.ok()) << listener_status;
  ASSERT_TRUE(runtime->Start().ok());

  asio::io_context client_io;
  beast::tcp_stream client(client_io);
  client.connect({asio::ip::make_address("127.0.0.1"),
                  static_cast<unsigned short>(listener->port())});
  beast::flat_buffer buffer;
  for (int i = 0; i < 2; ++i) {
    http::request<http::empty_body> request{http::verb::get, "/anything", 11};
    request.keep_alive(true);
    http::write(client, request);
    http::response<http::string_body> response;
    http::read(client, buffer, response);
    EXPECT_EQ(response.result(), http::status::ok);
    EXPECT_EQ(response.body(), "{\"ok\":true}");
  }

  http::request<http::string_body> oversized{http::verb::post, "/anything", 11};
  oversized.body() = "too-large";
  oversized.prepare_payload();
  http::write(client, oversized);
  http::response<http::string_body> rejected;
  http::read(client, buffer, rejected);
  EXPECT_EQ(rejected.result(), http::status::payload_too_large);

  beast::error_code ignored;
  client.socket().close(ignored);
  listener->Stop();
  runtime->Stop();
  runtime->Join();
}

}  // namespace
}  // namespace inferx::server::transport
