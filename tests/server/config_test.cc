// ServerConfig parse/validate tests (docs/server_design.md §10.1, §10.2,
// §14.1 step 4: every knob round-trips; invalid values are rejected).
#include <cstdlib>
#include <string>

#include "gtest/gtest.h"
#include "nlohmann/json.hpp"

#include "inferx/config/server_config.h"

namespace inferx::server {
namespace {

TEST(ServerConfigTest, DefaultsMatchDesignDoc) {
  const ServerConfig config;
  ASSERT_TRUE(config.Validate().ok());
  // §10.1 limits.
  EXPECT_EQ(config.max_connections, std::size_t{4096});
  EXPECT_EQ(config.max_inflight_requests, std::size_t{512});
  EXPECT_EQ(config.header_limit, std::size_t{16} * 1024);
  EXPECT_EQ(config.body_limit, std::size_t{4} * 1024 * 1024);
  EXPECT_EQ(config.max_pending_tokens, std::size_t{1024});
  EXPECT_EQ(config.max_output_bytes, std::size_t{8} * 1024 * 1024);
  EXPECT_EQ(config.max_write_batch_bytes, std::size_t{64} * 1024);
  EXPECT_EQ(config.max_read_ahead_bytes, std::size_t{16} * 1024);
  EXPECT_EQ(config.default_completion_max_tokens, 16u);
  EXPECT_FALSE(config.sse_early_headers);
  // §10.2 timeouts.
  EXPECT_EQ(config.header_read_timeout, std::chrono::seconds{10});
  EXPECT_EQ(config.body_read_timeout, std::chrono::seconds{30});
  EXPECT_EQ(config.keepalive_idle_timeout, std::chrono::seconds{60});
  EXPECT_EQ(config.queue_wait_timeout, std::chrono::seconds{30});
  EXPECT_EQ(config.request_timeout, std::chrono::seconds{600});
  EXPECT_EQ(config.stream_write_timeout, std::chrono::seconds{15});
  EXPECT_EQ(config.drain_timeout, std::chrono::seconds{30});
  EXPECT_EQ(config.cancel_grace, std::chrono::seconds{5});
  // Executors and api defaults.
  EXPECT_GE(config.io_threads, std::size_t{1});
  EXPECT_LE(config.io_threads, std::size_t{4});
  EXPECT_EQ(config.tokenizer_workers, std::size_t{2});
  EXPECT_EQ(config.tokenizer_queue_limit, std::size_t{64});
  EXPECT_EQ(config.api.host, "127.0.0.1");
  EXPECT_EQ(config.api.port, 8000);
  EXPECT_TRUE(config.api.api_keys.empty());
  EXPECT_FALSE(config.log_payloads);
}

TEST(ServerConfigTest, ValidateRejectsOutOfRangeValues) {
  const auto expect_invalid = [](const char* field, ServerConfig config) {
    const Status status = config.Validate();
    EXPECT_FALSE(status.ok()) << field;
    if (!status.ok()) {
      EXPECT_NE(status.message().find(field), std::string::npos)
          << status.message();
    }
  };

  ServerConfig config;
  ASSERT_TRUE(config.Validate().ok());

  ServerConfig bad_port_low = config;
  bad_port_low.api.port = 0;
  expect_invalid("api.port", bad_port_low);
  ServerConfig bad_port_high = config;
  bad_port_high.api.port = 65536;
  expect_invalid("api.port", bad_port_high);
  ServerConfig bad_host = config;
  bad_host.api.host = "";
  expect_invalid("api.host", bad_host);

  ServerConfig bad_threads = config;
  bad_threads.io_threads = 0;
  expect_invalid("io_threads", bad_threads);
  ServerConfig bad_workers = config;
  bad_workers.tokenizer_workers = 0;
  expect_invalid("tokenizer_workers", bad_workers);
  ServerConfig bad_queue = config;
  bad_queue.tokenizer_queue_limit = 0;
  expect_invalid("tokenizer_queue_limit", bad_queue);

  ServerConfig bad_conns = config;
  bad_conns.max_connections = 0;
  expect_invalid("max_connections", bad_conns);
  ServerConfig bad_inflight = config;
  bad_inflight.max_inflight_requests = 0;
  expect_invalid("max_inflight_requests", bad_inflight);
  ServerConfig bad_header = config;
  bad_header.header_limit = 128;
  expect_invalid("header_limit", bad_header);
  ServerConfig bad_body = config;
  bad_body.body_limit = 16;
  expect_invalid("body_limit", bad_body);
  ServerConfig bad_pending = config;
  bad_pending.max_pending_tokens = 0;
  expect_invalid("max_pending_tokens", bad_pending);

  // The write batch must fit inside the decoded-output cap.
  ServerConfig bad_cap = config;
  bad_cap.max_output_bytes = config.max_write_batch_bytes - 1;
  expect_invalid("max_output_bytes", bad_cap);

  ServerConfig bad_timeout = config;
  bad_timeout.request_timeout = std::chrono::seconds{0};
  expect_invalid("request_timeout", bad_timeout);
  ServerConfig bad_grace = config;
  bad_grace.cancel_grace = std::chrono::seconds{0};
  expect_invalid("cancel_grace", bad_grace);
}

TEST(ServerConfigTest, FromJsonAppliesOverridesAndKeepsDefaults) {
  const auto parsed = ServerConfig::FromJson(nlohmann::json::parse(R"({
    "api": {"host": "0.0.0.0", "port": 9999, "api_keys": ["k1", "k2"]},
    "max_connections": 128,
    "request_timeout": 120,
    "sse_early_headers": true,
    "log_payloads": true
  })"));
  ASSERT_TRUE(parsed.ok()) << parsed.status();
  EXPECT_EQ(parsed->api.host, "0.0.0.0");
  EXPECT_EQ(parsed->api.port, 9999);
  EXPECT_EQ(parsed->api.api_keys, (std::vector<std::string>{"k1", "k2"}));
  EXPECT_EQ(parsed->max_connections, std::size_t{128});
  EXPECT_EQ(parsed->request_timeout, std::chrono::seconds{120});
  EXPECT_TRUE(parsed->sse_early_headers);
  EXPECT_TRUE(parsed->log_payloads);
  // Absent keys keep defaults.
  EXPECT_EQ(parsed->max_inflight_requests, std::size_t{512});
  EXPECT_EQ(parsed->cancel_grace, std::chrono::seconds{5});
}

TEST(ServerConfigTest, FromJsonRejectsUnknownKeysAndBadTypes) {
  auto parsed = ServerConfig::FromJson(nlohmann::json::parse(R"({"conections": 8})"));
  EXPECT_FALSE(parsed.ok());  // Typo: unknown field.
  parsed = ServerConfig::FromJson(nlohmann::json::parse(R"({"io_threads": "four"})"));
  EXPECT_FALSE(parsed.ok());  // Wrong type.
  parsed = ServerConfig::FromJson(nlohmann::json::parse(R"({"request_timeout": -5})"));
  EXPECT_FALSE(parsed.ok());  // Negative duration.
  parsed = ServerConfig::FromJson(nlohmann::json::parse("[]"));
  EXPECT_FALSE(parsed.ok());  // Not an object.
  parsed = ServerConfig::FromJson(nlohmann::json::parse(R"({"api": {"port": 70000}})"));
  EXPECT_FALSE(parsed.ok());  // Range caught by Validate inside FromJson.
}

TEST(ServerConfigTest, FromJsonThenToJsonRoundTrips) {
  ServerConfig config;
  config.api.host = "10.0.0.1";
  config.api.port = 8443;
  config.api.api_keys = {"secret"};
  config.max_connections = 256;
  config.queue_wait_timeout = std::chrono::seconds{11};
  config.log_payloads = true;

  const auto reparsed = ServerConfig::FromJson(config.ToJson());
  ASSERT_TRUE(reparsed.ok()) << reparsed.status();
  EXPECT_EQ(reparsed->api.host, config.api.host);
  EXPECT_EQ(reparsed->api.port, config.api.port);
  EXPECT_EQ(reparsed->api.api_keys, config.api.api_keys);
  EXPECT_EQ(reparsed->max_connections, config.max_connections);
  EXPECT_EQ(reparsed->queue_wait_timeout, config.queue_wait_timeout);
  EXPECT_EQ(reparsed->log_payloads, config.log_payloads);
}

TEST(ServerConfigTest, EnvironmentMergesApiKey) {
  ServerConfig config;
  config.api.api_keys = {"from-flag"};
  ASSERT_EQ(0, ::setenv("INFERX_API_KEY", "from-env", /*replace=*/1));
  MergeEnvironmentOverrides(&config);
  ::unsetenv("INFERX_API_KEY");

  EXPECT_EQ(config.api.api_keys,
            (std::vector<std::string>{"from-flag", "from-env"}));

  // An empty value must not add an empty key.
  ServerConfig other;
  ASSERT_EQ(0, ::setenv("INFERX_API_KEY", "", 1));
  MergeEnvironmentOverrides(&other);
  ::unsetenv("INFERX_API_KEY");
  EXPECT_TRUE(other.api.api_keys.empty());
}

}  // namespace
}  // namespace inferx::server
