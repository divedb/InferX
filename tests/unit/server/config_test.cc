// ServerConfig unit tests: validation rejections, JSON file loading with
// unknown-key rejection, environment overrides, and CLI > env > file
// precedence (docs §12.2).
#include <cstdlib>
#include <fstream>
#include <string>

#include <gtest/gtest.h>

#include "inferx/server/server_config.h"

namespace inferx::server {
namespace {

ServerConfig ValidConfig() {
  ServerConfig config;
  config.model_id = "test-model";
  return config;
}

class ServerConfigTest : public ::testing::Test {
 protected:
  void TearDown() override {
    for (const char* var : {"INFERX_SERVER_HOST", "INFERX_SERVER_PORT",
                            "INFERX_SERVER_IO_THREADS", "INFERX_SERVER_MODEL",
                            "INFERX_SERVED_MODEL_NAMES"}) {
      unsetenv(var);
    }
  }

  std::string WriteConfigFile(const std::string& content) {
    const std::string path = ::testing::TempDir() + "inferx_server_test.json";
    { std::ofstream out(path, std::ios::trunc); out << content; }
    return path;
  }
};

TEST_F(ServerConfigTest, DefaultsValidate) {
  ServerConfig config = ValidConfig();
  EXPECT_TRUE(ValidateConfig(config).ok());
}

TEST_F(ServerConfigTest, MissingModelRejected) {
  ServerConfig config;
  EXPECT_FALSE(ValidateConfig(config).ok());
}

TEST_F(ServerConfigTest, RangeRejections) {
  {
    ServerConfig config = ValidConfig();
    config.io_threads = 0;
    EXPECT_FALSE(ValidateConfig(config).ok());
  }
  {
    ServerConfig config = ValidConfig();
    config.timeouts.keep_alive = std::chrono::milliseconds(0);
    EXPECT_FALSE(ValidateConfig(config).ok());
  }
  {
    ServerConfig config = ValidConfig();
    config.stream.engine_low_watermark_bytes =
        config.stream.engine_event_buffer_bytes + 1;
    EXPECT_FALSE(ValidateConfig(config).ok());
  }
  {
    ServerConfig config = ValidConfig();
    config.http.max_json_depth = 1;
    EXPECT_FALSE(ValidateConfig(config).ok());
  }
}

TEST_F(ServerConfigTest, TlsDisabledBuildRejectsEnabled) {
  ServerConfig config = ValidConfig();
  config.tls.enabled = true;
  const auto result = ValidateConfig(config);
  ASSERT_FALSE(result.ok());
  EXPECT_NE(result.error().message.find("tls"), std::string::npos);
}

TEST_F(ServerConfigTest, TlsCertAndKeyMustPair) {
  ServerConfig config = ValidConfig();
  config.tls.cert_file = "/tmp/does-not-exist.pem";
  config.tls.key_file = "/tmp/does-not-exist.key";
  config.tls.enabled = false;
  EXPECT_FALSE(ValidateConfig(config).ok());
}

TEST_F(ServerConfigTest, CorsWildcardWithCredentialsRejected) {
  ServerConfig config = ValidConfig();
  config.cors.enabled = true;
  config.cors.allow_credentials = true;
  config.cors.allow_origins = {"*"};
  EXPECT_FALSE(ValidateConfig(config).ok());
}

TEST_F(ServerConfigTest, AuthWithoutKeysRejected) {
  ServerConfig config = ValidConfig();
  config.auth.enabled = true;
  EXPECT_FALSE(ValidateConfig(config).ok());
}

TEST_F(ServerConfigTest, ConfigFileLoadsAndRejectsUnknownKeys) {
  const std::string good = WriteConfigFile(
      R"({"bind_host":"127.0.0.1","bind_port":9123,"io_threads":2,)"
      R"("model_id":"cfg-model","served_model_names":["a","b"],)"
      R"("http":{"max_body_bytes":1024},"timeouts":{"keep_alive_ms":250}})");
  auto loaded = LoadConfigFile(good);
  ASSERT_TRUE(loaded.ok()) << loaded.error().message;
  EXPECT_EQ(loaded.value().bind_port, 9123);
  EXPECT_EQ(loaded.value().io_threads, 2u);
  EXPECT_EQ(loaded.value().model_id, "cfg-model");
  EXPECT_EQ(loaded.value().served_model_names, (std::vector<std::string>{"a", "b"}));
  EXPECT_EQ(loaded.value().http.max_body_bytes, 1024u);
  EXPECT_EQ(loaded.value().timeouts.keep_alive, std::chrono::milliseconds(250));

  const std::string unknown = WriteConfigFile(R"({"totally_unknown": 1})");
  EXPECT_FALSE(LoadConfigFile(unknown).ok());

  const std::string bad_section = WriteConfigFile(R"({"tls": 5})");
  EXPECT_FALSE(LoadConfigFile(bad_section).ok());
}

TEST_F(ServerConfigTest, EnvOverridesApply) {
  setenv("INFERX_SERVER_HOST", "127.0.0.1", 1);
  setenv("INFERX_SERVER_PORT", "9999", 1);
  setenv("INFERX_SERVER_MODEL", "env-model", 1);
  setenv("INFERX_SERVED_MODEL_NAMES", "x,y", 1);
  ServerConfig config = ValidConfig();
  ASSERT_TRUE(ApplyEnvOverrides(config).ok());
  EXPECT_EQ(config.bind_host, "127.0.0.1");
  EXPECT_EQ(config.bind_port, 9999);
  EXPECT_EQ(config.model_id, "env-model");
  EXPECT_EQ(config.served_model_names, (std::vector<std::string>{"x", "y"}));

  setenv("INFERX_SERVER_PORT", "not-a-port", 1);
  EXPECT_FALSE(ApplyEnvOverrides(config).ok());
}

TEST_F(ServerConfigTest, CliBeatsEnvBeatsFile) {
  const std::string path = WriteConfigFile(
      R"({"bind_host":"10.0.0.1","bind_port":1111,"model_id":"file-model"})");
  setenv("INFERX_SERVER_PORT", "2222", 1);
  setenv("INFERX_SERVER_MODEL", "env-model", 1);
  const char* argv[] = {"prog", "--config", path.c_str(), "--port", "3333",
                        nullptr};
  auto loaded = LoadConfig(5, const_cast<char**>(argv));
  ASSERT_TRUE(loaded.ok()) << loaded.error().message;
  EXPECT_EQ(loaded.value().bind_host, "10.0.0.1");  // file value: not overridden
  EXPECT_EQ(loaded.value().bind_port, 3333);        // CLI beats env
  EXPECT_EQ(loaded.value().model_id, "env-model");  // env beats file
}

TEST_F(ServerConfigTest, UnknownCliFlagRejected) {
  const char* argv[] = {"prog", "--wat", nullptr};
  EXPECT_FALSE(LoadConfig(2, const_cast<char**>(argv)).ok());
}

}  // namespace
}  // namespace inferx::server
