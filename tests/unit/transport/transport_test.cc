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
