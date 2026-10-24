// HttpServer end-to-end suites over real loopback sockets (docs §13.2,
// §13.3, §13.6 Phase 1 acceptance): route behavior for /health, /livez,
// /readyz, /v1/models; HTTP/1.1 keep-alive, limits, and malformed-input
// handling; graceful drain/shutdown semantics.
#include <chrono>
#include <future>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include <boost/asio.hpp>
#include <nlohmann/json.hpp>

#include "http_test_util.h"
#include "inferx/server/http_server.h"
#include "inferx/server/routes.h"
#include "inferx/server/services.h"

namespace inferx::server {
namespace {

using testing::RawResponse;
using testing::TestHttpClient;

ServerConfig BaseConfig() {
  ServerConfig config;
  config.bind_host = "127.0.0.1";
  config.bind_port = 0;  // ephemeral
  config.model_id = "test-model";
  config.served_model_names = {"alias-a", "alias-b"};
  config.max_model_len = 8192;
  config.timeouts.keep_alive = std::chrono::milliseconds(300);
  config.timeouts.header_read = std::chrono::milliseconds(400);
  config.timeouts.body_read = std::chrono::milliseconds(400);
  config.timeouts.drain_grace = std::chrono::milliseconds(1000);
  config.timeouts.cancel_grace = std::chrono::milliseconds(300);
  config.io_threads = 2;
  return config;
}

/// \brief One fully wired server on its own io_context and threads.
///        Destruction drains and stops; tests construct as many as needed.
class ServerInstance {
 public:
  explicit ServerInstance(ServerConfig config) : config_(config) {
    catalog_ = std::make_shared<StaticModelCatalog>(
        StaticModelCatalog::FromConfig(config_.model_id,
                                       config_.served_model_names,
                                       config_.max_model_len));
    health_ = std::make_shared<HealthService>(nullptr);
    metrics_ = std::make_shared<CountingMetricsService>();
    AdmissionLimits limits;
    limits.max_body_bytes = config_.http.max_body_bytes;
    limits.max_active_requests = 64;
    admission_ = std::make_shared<LeaseAdmissionController>(limits, health_);
    router_ = BuildRouter(config_, catalog_, health_);
    middleware_ = BuildMiddleware(config_);
    server_ = std::make_unique<HttpServer>(config_, nullptr, catalog_, health_,
                                           metrics_, admission_, router_,
                                           middleware_);
    auto started = boost::asio::co_spawn(io_, server_->Start(),
                                         boost::asio::use_future);
    for (std::size_t i = 0; i < 2; ++i) {
      threads_.emplace_back([this] { io_.run(); });
    }
    auto bound = started.get();
    if (!bound.ok()) {
      work_.reset();
      for (auto& thread : threads_) {
        if (thread.joinable()) thread.join();
      }
      throw std::runtime_error("server failed to start: " +
                               bound.error().message);
    }
    port_ = bound.value().port;
  }

  // Keep the executor alive until Stop has completed. Checking stopped()
  // before co_spawn is racy, and restarting empty run() threads can let them
  // exit again before the shutdown coroutine is submitted.
  void StopAndJoin() {
    if (joined_) return;
    auto stopped = boost::asio::co_spawn(io_, server_->Stop(),
                                         boost::asio::use_future);
    stopped.get();
    work_.reset();
    for (auto& thread : threads_) {
      if (thread.joinable()) thread.join();
    }
    joined_ = true;
  }

  ~ServerInstance() { StopAndJoin(); }

  std::uint16_t port() const { return port_; }
  HealthService& health() { return *health_; }
  CountingMetricsService& metrics() { return *metrics_; }
  HttpServer& server() { return *server_; }
  boost::asio::io_context& io() { return io_; }

  TestHttpClient Connect() { return TestHttpClient("127.0.0.1", port_); }

  std::optional<RawResponse> Get(const std::string& target) {
    TestHttpClient client = Connect();
    if (!client.connected()) return std::nullopt;
    client.SendRequest("GET", target);
    return client.ReadResponse();
  }

 private:
  ServerConfig config_;
  std::shared_ptr<StaticModelCatalog> catalog_;
  std::shared_ptr<HealthService> health_;
  std::shared_ptr<CountingMetricsService> metrics_;
  std::shared_ptr<LeaseAdmissionController> admission_;
  std::shared_ptr<Router> router_;
  std::shared_ptr<MiddlewareChain> middleware_;
  std::unique_ptr<HttpServer> server_;
  boost::asio::io_context io_;
  boost::asio::executor_work_guard<boost::asio::io_context::executor_type> work_{
      boost::asio::make_work_guard(io_)};
  std::vector<std::thread> threads_;
  bool joined_ = false;
  std::uint16_t port_ = 0;
};

TEST(HttpServerRoutesTest, HealthIs200WhileServing) {
  ServerInstance server(BaseConfig());
  auto response = server.Get("/health");
  ASSERT_TRUE(response.has_value());
  EXPECT_EQ(response->status, 200);
  EXPECT_TRUE(response->body.empty());
}

TEST(HttpServerRoutesTest, LivezReportsOk) {
  ServerInstance server(BaseConfig());
  auto response = server.Get("/livez");
  ASSERT_TRUE(response.has_value());
  EXPECT_EQ(response->status, 200);
  EXPECT_EQ(nlohmann::json::parse(response->body)["status"], "ok");
}

TEST(HttpServerRoutesTest, ReadyzReflectsHealthState) {
  ServerInstance server(BaseConfig());
  auto ready = server.Get("/readyz");
  ASSERT_TRUE(ready.has_value());
  EXPECT_EQ(ready->status, 200);
  EXPECT_EQ(nlohmann::json::parse(ready->body)["status"], "ok");

  server.health().SetState(HealthState::kDraining, "draining for test");
  auto not_ready = server.Get("/readyz");
  ASSERT_TRUE(not_ready.has_value());
  EXPECT_EQ(not_ready->status, 503);
  EXPECT_EQ(nlohmann::json::parse(not_ready->body)["status"], "unavailable");
}

TEST(HttpServerRoutesTest, DrainingHealthGatesAdmissionWith503) {
  ServerInstance server(BaseConfig());
  server.health().SetState(HealthState::kDraining, "draining for test");
  auto response = server.Get("/v1/models");
  ASSERT_TRUE(response.has_value());
  EXPECT_EQ(response->status, 503);
  EXPECT_EQ(nlohmann::json::parse(response->body)["error"]["code"],
            "server_shutting_down");
}

TEST(HttpServerRoutesTest, ModelsListsConfiguredAliases) {
  ServerInstance server(BaseConfig());
  auto response = server.Get("/v1/models");
  ASSERT_TRUE(response.has_value());
  EXPECT_EQ(response->status, 200);
  const auto body = nlohmann::json::parse(response->body);
  ASSERT_EQ(body["object"], "list");
  ASSERT_TRUE(body["data"].is_array());
  ASSERT_EQ(body["data"].size(), 1u);
  EXPECT_EQ(body["data"][0]["id"], "alias-a");
  EXPECT_EQ(body["data"][0]["object"], "model");
  EXPECT_TRUE(body["data"][0]["created"].is_number());
  EXPECT_EQ(body["data"][0]["owned_by"], "inferx");
  EXPECT_EQ(body["data"][0]["metadata"]["max_model_len"], 8192);
}

TEST(HttpServerRoutesTest, UnknownPathIs404Envelope) {
  ServerInstance server(BaseConfig());
  auto response = server.Get("/v1/nope");
  ASSERT_TRUE(response.has_value());
  EXPECT_EQ(response->status, 404);
  const auto body = nlohmann::json::parse(response->body);
  EXPECT_EQ(body["error"]["type"], "invalid_request_error");
  EXPECT_EQ(body["error"]["code"], "not_found");
  EXPECT_FALSE(body["error"]["message"].get<std::string>().empty());
}

TEST(HttpServerRoutesTest, WrongMethodIs405WithAllow) {
  ServerInstance server(BaseConfig());
  TestHttpClient client = server.Connect();
  client.SendRequest("POST", "/v1/models", {}, "{}");
  auto response = client.ReadResponse();
  ASSERT_TRUE(response.has_value());
  EXPECT_EQ(response->status, 405);
  const std::string* allow = response->Header("Allow");
  ASSERT_NE(allow, nullptr);
  EXPECT_EQ(*allow, "GET");
  EXPECT_EQ(nlohmann::json::parse(response->body)["error"]["code"],
            "method_not_allowed");
}

TEST(HttpServerRoutesTest, OptionsIs405WithAllowHeader) {
  ServerInstance server(BaseConfig());
  TestHttpClient client = server.Connect();
  client.SendRequest("OPTIONS", "/v1/models");
  auto response = client.ReadResponse();
  ASSERT_TRUE(response.has_value());
  EXPECT_EQ(response->status, 405);
  const std::string* allow = response->Header("Allow");
  ASSERT_NE(allow, nullptr);
  EXPECT_EQ(*allow, "GET");
}

TEST(HttpServerHttp11Test, KeepAliveServesSequentialRequestsWithDistinctIds) {
  ServerInstance server(BaseConfig());
  TestHttpClient client = server.Connect();
  ASSERT_TRUE(client.connected());

  client.SendRequest("GET", "/livez");
  auto first = client.ReadResponse();
  ASSERT_TRUE(first.has_value());
  ASSERT_NE(first->Header("X-Request-Id"), nullptr);
  const std::string first_id = *first->Header("X-Request-Id");
  EXPECT_EQ(first_id.size(), 36u);

  client.SendRequest("GET", "/livez");
  auto second = client.ReadResponse();
  ASSERT_TRUE(second.has_value());
  ASSERT_NE(second->Header("X-Request-Id"), nullptr);
  EXPECT_EQ(second->status, 200);
  EXPECT_NE(*second->Header("X-Request-Id"), first_id);
}

TEST(HttpServerHttp11Test, ProvidedRequestIdIsEchoed) {
  ServerInstance server(BaseConfig());
  TestHttpClient client = server.Connect();
  client.SendRequest("GET", "/livez", {{"X-Request-Id", "client-echo-123"}});
  auto response = client.ReadResponse();
  ASSERT_TRUE(response.has_value());
  ASSERT_NE(response->Header("X-Request-Id"), nullptr);
  EXPECT_EQ(*response->Header("X-Request-Id"), "client-echo-123");
}

TEST(HttpServerHttp11Test, HeadRequestReturnsHeadersWithoutBody) {
  ServerInstance server(BaseConfig());
  TestHttpClient client = server.Connect();
  client.SendRequest("HEAD", "/livez");
  auto response = client.ReadResponse(/*expect_body=*/false);
  ASSERT_TRUE(response.has_value());
  EXPECT_EQ(response->status, 200);
  ASSERT_NE(response->Header("Content-Length"), nullptr);
  EXPECT_EQ(response->body.size(), 0u);
}

TEST(HttpServerHttp11Test, MalformedRequestLineIs400AndClose) {
  ServerInstance server(BaseConfig());
  TestHttpClient client = server.Connect();
  client.Send("GARBAGE\r\n\r\n");
  auto response = client.ReadResponse();
  ASSERT_TRUE(response.has_value());
  EXPECT_EQ(response->status, 400);
  EXPECT_EQ(nlohmann::json::parse(response->body)["error"]["code"],
            "invalid_json");
  EXPECT_TRUE(client.WaitForClose());
}

TEST(HttpServerHttp11Test, OversizedHeaderBlockIs431AndClose) {
  ServerInstance server(BaseConfig());
  const std::string big(40 * 1024, 'x');  // exceeds the 32 KiB header budget
  TestHttpClient client = server.Connect();
  client.Send("GET /livez HTTP/1.1\r\nHost: localhost\r\nX-Big: " + big +
              "\r\n\r\n");
  auto response = client.ReadResponse();
  ASSERT_TRUE(response.has_value());
  EXPECT_EQ(response->status, 431);
  EXPECT_TRUE(client.WaitForClose());
}

TEST(HttpServerHttp11Test, TooManyHeadersIsRejected) {
  ServerInstance server(BaseConfig());
  std::string request = "GET /livez HTTP/1.1\r\nHost: localhost\r\n";
  for (int i = 0; i < 200; ++i) {  // over the 100-header cap
    request += "X-Pad-" + std::to_string(i) + ": 1\r\n";
  }
  request += "\r\n";
  TestHttpClient client = server.Connect();
  client.Send(request);
  auto response = client.ReadResponse();
  ASSERT_TRUE(response.has_value());
  EXPECT_EQ(response->status, 431);
}

TEST(HttpServerHttp11Test, ContentLengthOverLimitIs413) {
  ServerConfig config = BaseConfig();
  config.http.max_body_bytes = 64;
  ServerInstance server(config);
  TestHttpClient client = server.Connect();
  // Send only the head with an oversized Content-Length: the server must
  // reject before waiting for the body bytes.
  client.Send("POST /v1/models HTTP/1.1\r\nHost: localhost\r\n"
              "Content-Type: application/json\r\n"
              "Content-Length: 200\r\n\r\n");
  auto response = client.ReadResponse();
  ASSERT_TRUE(response.has_value());
  EXPECT_EQ(response->status, 413);
  EXPECT_EQ(nlohmann::json::parse(response->body)["error"]["code"],
            "payload_too_large");
  EXPECT_TRUE(client.WaitForClose());
}

TEST(HttpServerHttp11Test, IdleKeepAliveClosesConnection) {
  ServerInstance server(BaseConfig());  // keep_alive = 300ms
  TestHttpClient client = server.Connect();
  client.SendRequest("GET", "/livez");
  ASSERT_TRUE(client.ReadResponse().has_value());
  EXPECT_TRUE(client.WaitForClose());
}

TEST(HttpServerHttp11Test, HeaderReadTimeoutClosesConnection) {
  ServerInstance server(BaseConfig());  // header_read = 400ms
  TestHttpClient client = server.Connect();
  client.Send("GET /livez HT");  // partial request line, then silence
  EXPECT_TRUE(client.WaitForClose());
}

TEST(HttpServerHttp11Test, PrematureEofMidBodyDoesNotCrash) {
  ServerInstance server(BaseConfig());
  TestHttpClient client = server.Connect();
  client.Send("POST /v1/models HTTP/1.1\r\nHost: localhost\r\n"
              "Content-Length: 100\r\n\r\nshort");
  client.Close();
  // The server must discard the truncated request without dispatching.
  TestHttpClient follow_up = server.Connect();
  follow_up.SendRequest("GET", "/livez");
  auto response = follow_up.ReadResponse();
  ASSERT_TRUE(response.has_value());
  EXPECT_EQ(response->status, 200);
}

TEST(HttpServerLifecycleTest, MaxRequestsPerConnectionClosesAfterLimit) {
  ServerConfig config = BaseConfig();
  config.http.max_requests_per_connection = 2;
  ServerInstance server(config);
  TestHttpClient client = server.Connect();
  for (int i = 0; i < 2; ++i) {
    client.SendRequest("GET", "/livez");
    auto response = client.ReadResponse();
    ASSERT_TRUE(response.has_value());
    EXPECT_EQ(response->status, 200);
  }
  EXPECT_TRUE(client.WaitForClose());
}

TEST(HttpServerLifecycleTest, BeginDrainClosesIdleConnectionsAndRefusesNew) {
  ServerInstance server(BaseConfig());
  TestHttpClient client = server.Connect();
  client.SendRequest("GET", "/livez");
  ASSERT_TRUE(client.ReadResponse().has_value());

  server.server().BeginDrain(DrainPolicy::kFinishActive);
  // The idle session is closed out of its keep-alive read.
  EXPECT_TRUE(client.WaitForClose());

  // New connections fail: the listener is closed.
  TestHttpClient late = TestHttpClient("127.0.0.1", server.port());
  EXPECT_FALSE(late.connected());
}

TEST(HttpServerLifecycleTest, InFlightRequestFinishesDuringDrain) {
  ServerConfig config = BaseConfig();
  config.timeouts.keep_alive = std::chrono::milliseconds(5000);
  ServerInstance server(config);
  TestHttpClient client = server.Connect();
  client.SendRequest("GET", "/v1/models");
  server.server().BeginDrain(DrainPolicy::kFinishActive);
  auto response = client.ReadResponse();
  // The request raced the drain: served normally, rejected with 503, or
  // cleanly cut off — all are valid terminal outcomes (docs §6.7).
  if (response.has_value()) {
    EXPECT_TRUE(response->status == 200 || response->status == 503)
        << "unexpected status " << response->status;
  }
  EXPECT_TRUE(client.WaitForClose());
}

TEST(HttpServerLifecycleTest, StopCompletesAndDrainsIo) {
  ServerInstance server(BaseConfig());
  server.server().BeginDrain(DrainPolicy::kFinishActive);
  server.StopAndJoin();
  EXPECT_FALSE(server.server().serving());
  EXPECT_EQ(server.server().active_connections(), 0u);

  // Releasing the fixture work guard lets run() exhaust its work naturally.
  EXPECT_TRUE(server.io().stopped());
}

TEST(HttpServerLifecycleTest, HealthStateTransitionsToStopped) {
  ServerInstance server(BaseConfig());
  server.server().BeginDrain(DrainPolicy::kFinishActive);
  server.StopAndJoin();
  EXPECT_EQ(server.health().Snapshot().state, HealthState::kStopped);
}

TEST(HttpServerLifecycleTest, StopAfterDrainHasCompleted) {
  ServerInstance server(BaseConfig());
  server.server().BeginDrain(DrainPolicy::kFinishActive);
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
  while (server.health().Snapshot().state != HealthState::kStopped &&
         std::chrono::steady_clock::now() < deadline) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  ASSERT_EQ(server.health().Snapshot().state, HealthState::kStopped);

  // Queue Stop only after drain has finished, when the old fixture could
  // lose all run() threads. The work guard must keep the executor available.
  auto stopped = boost::asio::co_spawn(server.io(), server.server().Stop(),
                                       boost::asio::use_future);
  const auto ready = stopped.wait_for(std::chrono::seconds(2));
  EXPECT_EQ(ready, std::future_status::ready);
  if (ready == std::future_status::ready) stopped.get();
  server.StopAndJoin();
  EXPECT_TRUE(server.io().stopped());
  server.StopAndJoin();  // Explicit shutdown followed by teardown is safe.
}

TEST(HttpServerLifecycleTest, ConcurrentClientsAreServed) {
  ServerInstance server(BaseConfig());
  TestHttpClient a = server.Connect();
  TestHttpClient b = server.Connect();
  a.SendRequest("GET", "/health");
  b.SendRequest("GET", "/health");
  auto ra = a.ReadResponse();
  auto rb = b.ReadResponse();
  ASSERT_TRUE(ra.has_value());
  ASSERT_TRUE(rb.has_value());
  EXPECT_EQ(ra->status, 200);
  EXPECT_EQ(rb->status, 200);
}

TEST(HttpServerLifecycleTest, MetricsObserveRequests) {
  ServerInstance server(BaseConfig());
  ASSERT_TRUE(server.Get("/livez").has_value());
  ASSERT_TRUE(server.Get("/v1/models").has_value());
  EXPECT_GE(server.metrics().http_requests_total(), 2u);
}

TEST(HttpServerLifecycleTest, ServerRejectsInvalidConfigAtStart) {
  ServerConfig config = BaseConfig();
  config.model_id.clear();
  config.served_model_names.clear();  // invalid: neither model id nor aliases
  ASSERT_ANY_THROW({ ServerInstance server(config); });
}

}  // namespace
}  // namespace inferx::server
