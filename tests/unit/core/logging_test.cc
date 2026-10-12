#include <cstdlib>
#include <fstream>
#include <map>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "gtest/gtest.h"
#include "inferx/core/logging.h"
#include "inferx/core/status.h"

namespace inferx::core {
namespace {

// Tests share one process, and InitializeLogging() may succeed exactly once in
// it. Every suite here is written to that constraint: the rejection tests run
// before InitializeLogging's one success, which is the last test in the file.
// A rejected call does not trip the once-only latch (validation happens before
// any global state changes), so the rejection tests may all run first.

// -- ParseLogLevel ------------------------------------------------------------

TEST(ParseLogLevel, ParsesTheThreeLevels) {
  EXPECT_EQ(ParseLogLevel("info").value(), LogLevel::kInfo);
  EXPECT_EQ(ParseLogLevel("warning").value(), LogLevel::kWarning);
  EXPECT_EQ(ParseLogLevel("error").value(), LogLevel::kError);
}

TEST(ParseLogLevel, RejectsUnknownNames) {
  EXPECT_FALSE(ParseLogLevel("verbose").ok());
  EXPECT_FALSE(ParseLogLevel("INFO").ok());  // Case-sensitive, like the docs.
  EXPECT_FALSE(ParseLogLevel("fatal").ok());  // FATAL is never a threshold.
  EXPECT_FALSE(ParseLogLevel("").ok());
}

// -- LoggingConfigFromEnvironment ---------------------------------------------

class LoggingEnvTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Each case must see a known environment; the loader reads all three
    // variables, so all three are saved and cleared.
    for (const char* name : {"INFERX_CONFIG_FILE", "INFERX_LOG_LEVEL", "INFERX_VLOG"}) {
      const char* value = std::getenv(name);
      saved_[name] = value == nullptr ? std::nullopt : std::optional<std::string>(value);
      unsetenv(name);
    }
  }

  void TearDown() override {
    for (const auto& [name, value] : saved_) {
      if (value.has_value()) setenv(name.c_str(), value->c_str(), 1);
    }
  }

  // Writes `contents` to a uniquely named file under the gtest temp dir.
  std::string WriteTempFile(const std::string& name, const std::string& contents) {
    const std::string path = std::string(testing::TempDir()) + "/" + name;
    std::ofstream out(path, std::ios::trunc);
    out << contents;
    return path;
  }

  std::map<std::string, std::optional<std::string>> saved_;
};

TEST_F(LoggingEnvTest, DefaultsWhenEnvironmentIsClean) {
  const StatusOr<LoggingConfig> config = LoggingConfigFromEnvironment();
  ASSERT_TRUE(config.ok()) << config.status();
  EXPECT_EQ(config->min_level, LogLevel::kInfo);
  EXPECT_EQ(config->vlog_verbosity, 0);
  EXPECT_TRUE(config->vlog_modules.empty());
  EXPECT_TRUE(config->log_file.empty());
}

TEST_F(LoggingEnvTest, EnvironmentVariablesApply) {
  setenv("INFERX_LOG_LEVEL", "warning", 1);
  setenv("INFERX_VLOG", "3", 1);
  const StatusOr<LoggingConfig> config = LoggingConfigFromEnvironment();
  ASSERT_TRUE(config.ok()) << config.status();
  EXPECT_EQ(config->min_level, LogLevel::kWarning);
  EXPECT_EQ(config->vlog_verbosity, 3);
}

// The loader fails loudly rather than silently falling back to defaults, so a
// typo in a variable surfaces at startup instead of hiding the setting.
TEST_F(LoggingEnvTest, BadEnvironmentValuesAreRejected) {
  setenv("INFERX_LOG_LEVEL", "loud", 1);
  const StatusOr<LoggingConfig> loud = LoggingConfigFromEnvironment();
  ASSERT_FALSE(loud.ok());
  EXPECT_TRUE(absl::StrContains(loud.status().message(), "INFERX_LOG_LEVEL"));

  unsetenv("INFERX_LOG_LEVEL");
  setenv("INFERX_VLOG", "-2", 1);
  const StatusOr<LoggingConfig> negative = LoggingConfigFromEnvironment();
  ASSERT_FALSE(negative.ok());
  EXPECT_TRUE(absl::StrContains(negative.status().message(), "INFERX_VLOG"));
}

TEST_F(LoggingEnvTest, ReadsYamlConfigFile) {
  const std::string path = WriteTempFile(
      "inferx_logging_yaml_ok.yaml",
      "logging:\n"
      "  level: error\n"
      "  vlog: 2\n"
      "  vlog_modules:\n"
      "    model_runner: 3\n"
      "    core: 1\n"
      "  file: /tmp/inferx-from-yaml.log\n");
  setenv("INFERX_CONFIG_FILE", path.c_str(), 1);

  const StatusOr<LoggingConfig> config = LoggingConfigFromEnvironment();
  ASSERT_TRUE(config.ok()) << config.status();
  EXPECT_EQ(config->min_level, LogLevel::kError);
  EXPECT_EQ(config->vlog_verbosity, 2);
  EXPECT_EQ(config->vlog_modules,
            (std::vector<std::pair<std::string, int>>{{"model_runner", 3}, {"core", 1}}));
  EXPECT_EQ(config->log_file, "/tmp/inferx-from-yaml.log");
}

TEST_F(LoggingEnvTest, EnvironmentLevelOverridesFile) {
  const std::string path =
      WriteTempFile("inferx_logging_yaml_precedence.yaml", "logging:\n  level: error\n");
  setenv("INFERX_CONFIG_FILE", path.c_str(), 1);
  setenv("INFERX_LOG_LEVEL", "info", 1);

  const StatusOr<LoggingConfig> config = LoggingConfigFromEnvironment();
  ASSERT_TRUE(config.ok()) << config.status();
  EXPECT_EQ(config->min_level, LogLevel::kInfo);
}

TEST_F(LoggingEnvTest, EmptyFileYieldsDefaults) {
  const std::string path = WriteTempFile("inferx_logging_yaml_empty.yaml", "");
  setenv("INFERX_CONFIG_FILE", path.c_str(), 1);
  const StatusOr<LoggingConfig> config = LoggingConfigFromEnvironment();
  ASSERT_TRUE(config.ok()) << config.status();
  EXPECT_EQ(config->min_level, LogLevel::kInfo);
}

TEST_F(LoggingEnvTest, MissingConfigFileIsRejected) {
  setenv("INFERX_CONFIG_FILE", "/no/such/inferx-config.yaml", 1);
  EXPECT_FALSE(LoggingConfigFromEnvironment().ok());
}

TEST_F(LoggingEnvTest, MalformedYamlIsRejected) {
  const std::string path =
      WriteTempFile("inferx_logging_yaml_bad.yaml", "logging: {level: info\n");
  setenv("INFERX_CONFIG_FILE", path.c_str(), 1);
  const StatusOr<LoggingConfig> config = LoggingConfigFromEnvironment();
  ASSERT_FALSE(config.ok());
  EXPECT_TRUE(absl::StrContains(config.status().message(), "cannot parse"));
}

TEST_F(LoggingEnvTest, UnknownLoggingKeyIsRejected) {
  const std::string path =
      WriteTempFile("inferx_logging_yaml_unknown.yaml", "logging:\n  verbosity: 2\n");
  setenv("INFERX_CONFIG_FILE", path.c_str(), 1);
  const StatusOr<LoggingConfig> config = LoggingConfigFromEnvironment();
  ASSERT_FALSE(config.ok());
  EXPECT_TRUE(absl::StrContains(config.status().message(), "verbosity"));
}

TEST_F(LoggingEnvTest, TopLevelMustBeAMap) {
  const std::string path = WriteTempFile("inferx_logging_yaml_seq.yaml", "- one\n- two\n");
  setenv("INFERX_CONFIG_FILE", path.c_str(), 1);
  const StatusOr<LoggingConfig> config = LoggingConfigFromEnvironment();
  ASSERT_FALSE(config.ok());
  EXPECT_TRUE(absl::StrContains(config.status().message(), "top level must be a map"));
}

TEST_F(LoggingEnvTest, BadModuleEntriesAreRejected) {
  {
    const std::string path = WriteTempFile(
        "inferx_logging_yaml_seq_modules.yaml", "logging:\n  vlog_modules: [a, b]\n");
    setenv("INFERX_CONFIG_FILE", path.c_str(), 1);
    EXPECT_FALSE(LoggingConfigFromEnvironment().ok());  // Not a map.
  }
  {
    const std::string path =
        WriteTempFile("inferx_logging_yaml_empty_pattern.yaml", "logging:\n  vlog_modules:\n    '': 1\n");
    setenv("INFERX_CONFIG_FILE", path.c_str(), 1);
    EXPECT_FALSE(LoggingConfigFromEnvironment().ok());  // Empty pattern.
  }
  {
    const std::string path =
        WriteTempFile("inferx_logging_yaml_bad_module_v.yaml", "logging:\n  vlog_modules:\n    foo: -1\n");
    setenv("INFERX_CONFIG_FILE", path.c_str(), 1);
    EXPECT_FALSE(LoggingConfigFromEnvironment().ok());  // Negative verbosity.
  }
}

// -- InitializeLogging ---------------------------------------------------------

// These rejection tests rely on running before the successful initialization
// below; see the comment at the top of the file.

TEST(InitializeLogging, RejectsNegativeVerbosity) {
  LoggingConfig config;
  config.vlog_verbosity = -1;
  EXPECT_FALSE(InitializeLogging(config).ok());
}

TEST(InitializeLogging, RejectsBadModuleEntries) {
  LoggingConfig empty_pattern;
  empty_pattern.vlog_modules = {{"", 1}};
  EXPECT_FALSE(InitializeLogging(empty_pattern).ok());

  LoggingConfig negative;
  negative.vlog_modules = {{"model_runner", -1}};
  EXPECT_FALSE(InitializeLogging(negative).ok());
}

TEST(InitializeLogging, RejectsUnopenableLogFile) {
  LoggingConfig config;
  config.log_file = std::string(testing::TempDir()) + "/no_such_dir/inferx.log";
  const Status status = InitializeLogging(config);
  ASSERT_FALSE(status.ok());
  EXPECT_TRUE(absl::StrContains(status.message(), "cannot open log file"));
}

// The process's one successful initialization. Must stay last: it trips the
// once-only latch, so every later (or rerun) call would only see rejection.
TEST(InitializeLogging, InitializesOnceAppliesVlogThresholdsAndWritesFile) {
  const std::string log_path = std::string(testing::TempDir()) + "/inferx_logging_test.log";
  std::remove(log_path.c_str());

  LoggingConfig config;
  config.min_level = LogLevel::kInfo;
  config.vlog_verbosity = 1;
  // Raises just this file above the global threshold, exercising the
  // per-module path; the pattern matches this file's basename.
  config.vlog_modules = {{"logging_test*", 2}};
  config.log_file = log_path;
  ASSERT_TRUE(InitializeLogging(config).ok()) << "must be the first successful call";

  // Once-only: a second call is rejected instead of reconfiguring the sinks.
  EXPECT_FALSE(InitializeLogging(LoggingConfig{}).ok());

  INFERX_LOG(INFO) << "plain info line";
  INFERX_VLOG(1) << "vlog at global threshold";   // 1 <= 1: passes the global gate.
  INFERX_VLOG(2) << "vlog via module override";   // 2 <= 2: passes only via the module.
  INFERX_VLOG(3) << "vlog above every threshold";  // 3 > 2: dropped.
  INFERX_LOG(WARNING) << "flush marker";  // >= WARNING flushes the sink's file.

  std::ifstream in(log_path);
  ASSERT_TRUE(in.good()) << "log file was not created";
  const std::string contents((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
  EXPECT_TRUE(absl::StrContains(contents, "plain info line"));
  EXPECT_TRUE(absl::StrContains(contents, "vlog at global threshold"));
  EXPECT_TRUE(absl::StrContains(contents, "vlog via module override"));
  EXPECT_FALSE(absl::StrContains(contents, "vlog above every threshold"));
  EXPECT_TRUE(absl::StrContains(contents, "flush marker"));
}

}  // namespace
}  // namespace inferx::core
