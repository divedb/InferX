#include <gtest/gtest.h>

#include "inferx/server/gateway/gateway_server.h"

namespace inferx::server::gateway {
namespace {

GatewayServerConfig ValidConfig() {
  GatewayServerConfig config;
  config.scheduler_endpoint = "dns:///scheduler:50051";
  config.tokenizer_directory = "/models/example";
  config.model_id = "example";
  return config;
}

TEST(GatewayConfigTest, AcceptsHostOnlyRemoteConfiguration) {
  EXPECT_TRUE(ValidateGatewayServerConfig(ValidConfig()).ok());
}

TEST(GatewayConfigTest, RequiresRemoteScheduler) {
  auto config = ValidConfig();
  config.scheduler_endpoint.clear();
  EXPECT_EQ(ValidateGatewayServerConfig(config).code(),
            absl::StatusCode::kInvalidArgument);
}

TEST(GatewayConfigTest, RequiresCpuTokenizerArtifacts) {
  auto config = ValidConfig();
  config.tokenizer_directory.clear();
  EXPECT_EQ(ValidateGatewayServerConfig(config).code(),
            absl::StatusCode::kInvalidArgument);
}

TEST(GatewayConfigTest, RejectsInvalidLimitsAndCredentials) {
  auto config = ValidConfig();
  config.max_active_requests = 0;
  EXPECT_EQ(ValidateGatewayServerConfig(config).code(),
            absl::StatusCode::kInvalidArgument);
  config = ValidConfig();
  config.api_key_sha256 = {"not-a-sha256-digest"};
  EXPECT_EQ(ValidateGatewayServerConfig(config).code(),
            absl::StatusCode::kInvalidArgument);
}

}  // namespace
}  // namespace inferx::server::gateway
