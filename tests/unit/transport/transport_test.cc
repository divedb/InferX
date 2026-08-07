#include <gtest/gtest.h>

#include <boost/asio.hpp>
#include <boost/beast.hpp>
#include <folly/CancellationToken.h>
#include <folly/coro/BlockingWait.h>
#include <folly/coro/Task.h>
#include <folly/executors/CPUThreadPoolExecutor.h>

#include <memory>
#include <string>
#include <thread>
#include <utility>

#include "inferx/server/transport/beast_folly_adapter.h"
#include "inferx/server/transport/beast_listener.h"
#include "inferx/server/transport/io_runtime.h"
#include "inferx/server/transport/routes.h"
#include "inferx/server/transport/sse_writer.h"
#include "inferx/server/api/error_mapping.h"
#include "inferx/server/request/request_context.h"
#include "inferx/server/request/request_manager.h"
#include "inferx/server/streaming/event_buffer.h"
#include "inferx/server/streaming/event_router.h"
#include "inferx/server/admission/admission_controller.h"
#include "inferx/server/auth/api_key_store.h"
#include "inferx/server/auth/rbac.h"
#include "inferx/server/model_registry/registry.h"

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
      HttpRequest request, ResponseWriter& writer,
      folly::CancellationToken cancellation) override {
    HttpResponse response{http::status::ok, request.version()};
    response.set(http::field::content_type, "application/json");
    response.keep_alive(request.keep_alive());
    response.body() = "{\"ok\":true}";
    response.prepare_payload();
    (void)co_await writer.WriteResponse(std::move(response), cancellation);
  }
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
      routes.Handle(std::move(wrong_method), writer, {}));
  EXPECT_EQ(writer.status, 405);

  CapturingWriter missing_writer;
  HttpRequest missing{http::verb::get, "/missing", 11};
  folly::coro::blockingWait(
      routes.Handle(std::move(missing), missing_writer, {}));
  EXPECT_EQ(missing_writer.status, 404);
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
                           ::inferx::server::model_registry::ModelState::kReady)
                  .ok());
  auto resolved = registry.Resolve("current");
  ASSERT_TRUE(resolved.ok()) << resolved.status();
  EXPECT_EQ(resolved->version, "v1");
  EXPECT_EQ(registry.ReadyModels().size(), 1);
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
