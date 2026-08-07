#include <gtest/gtest.h>

#include <boost/asio.hpp>
#include <boost/beast.hpp>
#include <folly/CancellationToken.h>
#include <folly/coro/BlockingWait.h>
#include <folly/coro/Task.h>
#include <folly/executors/CPUThreadPoolExecutor.h>
#include <folly/futures/ManualTimekeeper.h>

#include <memory>
#include <string>
#include <thread>
#include <utility>

#include "inferx/server/transport/beast_folly_adapter.h"
#include "inferx/server/transport/beast_listener.h"
#include "inferx/server/transport/io_runtime.h"
#include "inferx/server/transport/routes.h"
#include "inferx/server/transport/sse_writer.h"
#include "inferx/api/openai.h"
#include "inferx/server/api/error_mapping.h"
#include "inferx/server/request/request_context.h"
#include "inferx/server/request/request_manager.h"
#include "inferx/server/request/request_id.h"
#include "inferx/server/streaming/event_buffer.h"
#include "inferx/server/streaming/backpressure_controller.h"
#include "inferx/server/streaming/event_router.h"
#include "inferx/server/admission/admission_controller.h"
#include "inferx/server/admission/capacity_policy.h"
#include "inferx/server/auth/api_key_store.h"
#include "inferx/server/auth/authenticator.h"
#include "inferx/server/auth/rbac.h"
#include "inferx/server/model_registry/registry.h"
#include "inferx/server/middleware/authentication.h"
#include "inferx/server/handlers/health_handler.h"
#include "inferx/server/tokenization/tokenization_service.h"
#include "inferx/server/coroutine/deadline.h"

namespace inferx::server::transport {
namespace {

namespace asio = boost::asio;
namespace beast = boost::beast;
namespace http = beast::http;
using tcp = asio::ip::tcp;

class CapturingWriter final : public ResponseWriter {
 public:
  folly::coro::Task<Status> WriteResponse(
      HttpResponse response, folly::CancellationToken) override {
    status = response.result_int();
    body = std::move(response.body());
    co_return OkStatus();
  }
  folly::coro::Task<Status> StartStream(
      HttpStreamHead response, folly::CancellationToken) override {
    status = response.result_int();
    co_return OkStatus();
  }
  folly::coro::Task<Status> Write(
      std::string data, folly::CancellationToken) override {
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

class ScopeGuard final : public RouteGuard {
 public:
  folly::coro::Task<Status> Check(
      const HttpRequest&, const RouteMetadata& metadata,
      RequestContext&,
      folly::CancellationToken) override {
    seen_scope = metadata.required_scope;
    if (deny) co_return absl::PermissionDeniedError("scope denied");
    co_return OkStatus();
  }
  bool deny = false;
  std::string seen_scope;
};

TEST(BeastFollyAdapterTest, CompletesAndReturnsToFollyExecutor) {
  asio::io_context io;
  auto work = asio::make_work_guard(io);
  std::thread io_thread([&] { io.run(); });
  folly::CPUThreadPoolExecutor executor(1);

  auto task = [&]() -> folly::coro::Task<StatusOr<size_t>> {
    co_return co_await AwaitAsio<size_t>(
        io.get_executor(),
        [](auto completion) { completion({}, 17); }, [] {});
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
  ASSERT_TRUE(routes.Add(http::verb::get, "/ready",
                         std::make_shared<OkHandler>()).ok());

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
  ASSERT_TRUE(routes.Add(http::verb::post, "/completion",
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
      ::inferx::server::api::MapStatus(
          ResourceExhaustedError("quota \"full\""), "max_tokens");
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
  EXPECT_FALSE(::inferx::api::ParseCompletionRequest(
                   R"({"model":"m","prompt":"p","n":2})")
                   .ok());
  EXPECT_EQ(::inferx::api::ParseCompletionRequest(
                R"({"model":"m","prompt":"p","top_k":40})")
                .status()
                .code(),
            absl::StatusCode::kUnimplemented);
  EXPECT_EQ(::inferx::api::ParseCompletionRequest(
                R"({"model":"m","prompt":"p","repetition_penalty":1.1})")
                .status()
                .code(),
            absl::StatusCode::kUnimplemented);
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
  auto result = folly::coro::blockingWait(
      ::inferx::server::coroutine::WithDeadline(
          success(), std::chrono::steady_clock::now() +
                         std::chrono::seconds(1),
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
  result = folly::coro::blockingWait(
      ::inferx::server::coroutine::WithDeadline(
          expired(), std::chrono::steady_clock::now() -
                         std::chrono::milliseconds(1),
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
  ASSERT_TRUE(manager.Cancel(
                          "req_manager",
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
  const std::string first =
      ::inferx::server::request::GenerateRequestId(epoch);
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
    ASSERT_TRUE(manager.Cancel(
                           id, ::inferx::server::request::CancellationReason::
                                   kAdministrative)
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
  CapacityPolicy policy({.max_queued_requests = 2,
                         .max_event_buffer_bytes = 100,
                         .overload_retry_after = std::chrono::milliseconds(250),
                         .unavailable_retry_after =
                             std::chrono::milliseconds(500)});

  auto decision = policy.Evaluate({});
  EXPECT_EQ(decision.reason, AdmissionReason::kSchedulerUnavailable);
  EXPECT_EQ(decision.retry_after, std::chrono::milliseconds(500));

  decision = policy.Evaluate({.scheduler_healthy = true});
  EXPECT_EQ(decision.reason, AdmissionReason::kModelUnavailable);

  decision = policy.Evaluate({.scheduler_healthy = true,
                              .model_ready = true,
                              .queued_requests = 2});
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
  EXPECT_TRUE(limiter.Consume("tenant_rate", 1,
                             start + std::chrono::seconds(1))
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
  EXPECT_TRUE(::inferx::server::auth::Authorize(*found, "completions:write").ok());
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
  ASSERT_TRUE(
      store
          .AddHash(
              "2bb80d537b1da3e38bd30361aa855686bde0eacd7162fef6a25fe97bf527a25b",
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
  ASSERT_TRUE(
      store
          ->AddHash(
              "2bb80d537b1da3e38bd30361aa855686bde0eacd7162fef6a25fe97bf527a25b",
              principal)
          .ok());
  auto authenticator =
      std::make_shared<::inferx::server::auth::ApiKeyAuthenticator>(store.get());
  auto guard = std::make_shared<
      ::inferx::server::middleware::BearerRouteGuard>(authenticator);
  Routes routes(guard);
  auto invoke_handler = std::make_shared<OkHandler>();
  ASSERT_TRUE(routes.Add(http::verb::post, "/invoke",
                         {.authentication_required = true,
                          .required_scope = "inference.invoke"},
                         invoke_handler)
                  .ok());
  ASSERT_TRUE(routes.Add(http::verb::get, "/health",
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
  EXPECT_TRUE(
      invoke_handler->last_context.scopes.contains("inference.invoke"));

  CapturingWriter health_writer;
  HttpRequest health{http::verb::get, "/health", 11};
  folly::coro::blockingWait(
      routes.Handle(std::move(health), {}, health_writer, {}));
  EXPECT_EQ(health_writer.status, 200);
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
  ASSERT_TRUE(registry.SetState(
                           "model", "v1",
                           ::inferx::server::model_registry::ModelState::kWarming)
                  .ok());
  ASSERT_TRUE(registry.SetState(
                           "model", "v1",
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
  ASSERT_TRUE(
      registry.SetState("embedding", "v2", ModelState::kLoading).ok());
  ASSERT_TRUE(
      registry.SetState("embedding", "v2", ModelState::kWarming).ok());
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
  event.finish_reason =
      ::inferx::server::scheduler_client::FinishReason::kStop;
  event.usage = {.prompt_tokens = 4, .completion_tokens = 2};
  event.embeddings.push_back({.index = 0, .values = {0.25f, -0.5f}});
  EXPECT_EQ(event.usage.prompt_tokens, 4);
  EXPECT_EQ(event.embeddings.front().values.size(), 2);
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
  folly::coro::blockingWait(ready.Handle(
      std::move(first), {}, unavailable, folly::CancellationToken{}));
  EXPECT_EQ(unavailable.status, 503);

  state.SetReadyModelAvailable(true);
  CapturingWriter available;
  HttpRequest second{http::verb::get, "/health/ready", 11};
  folly::coro::blockingWait(ready.Handle(
      std::move(second), {}, available, folly::CancellationToken{}));
  EXPECT_EQ(available.status, 200);
}

TEST(BeastListenerTest, ServesSequentialKeepAliveRequests) {
  auto runtime_result = IoRuntime::Create({1, 1, 1});
  ASSERT_TRUE(runtime_result.ok()) << runtime_result.status();
  auto runtime = std::move(*runtime_result);
  BeastListenerConfig config;
  config.max_request_bytes = 4;
  auto listener_result = BeastListener::Create(
      runtime.get(), config, std::make_shared<OkHandler>());
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
