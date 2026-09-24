#include <cstdlib>
#include <map>
#include <optional>
#include <string>

#include "absl/strings/str_cat.h"
#include "gtest/gtest.h"
#include "inferx/core/logging.h"
#include "inferx/core/status.h"

namespace inferx::core {
namespace {

class LoggingTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Tests run in one process; save and clear both variables so each case
    // sees a known environment.
    for (const char* name : {"INFERX_LOG_LEVEL", "INFERX_VLOG"}) {
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

  std::map<std::string, std::optional<std::string>> saved_;
};

TEST(ParseLogLevel, ParsesTheThreeLevels) {
  EXPECT_EQ(ParseLogLevel("info").value(), LogLevel::kInfo);
  EXPECT_EQ(ParseLogLevel("warning").value(), LogLevel::kWarning);
  EXPECT_EQ(ParseLogLevel("error").value(), LogLevel::kError);
}

TEST(ParseLogLevel, RejectsUnknownNames) {
  EXPECT_FALSE(ParseLogLevel("verbose").ok());
  EXPECT_FALSE(ParseLogLevel("INFO").ok());  // Case-sensitive, like the env docs say.
  EXPECT_FALSE(ParseLogLevel("").ok());
}

TEST_F(LoggingTest, EnvDefaultsApplyWhenUnset) {
  const auto config = LoggingConfigFromEnv();
  EXPECT_EQ(config.min_level, LogLevel::kInfo);
  EXPECT_EQ(config.vlog_verbosity, 0);
}

TEST_F(LoggingTest, EnvOverridesApply) {
  setenv("INFERX_LOG_LEVEL", "warning", 1);
  setenv("INFERX_VLOG", "3", 1);
  const auto config = LoggingConfigFromEnv();
  EXPECT_EQ(config.min_level, LogLevel::kWarning);
  EXPECT_EQ(config.vlog_verbosity, 3);
}

TEST_F(LoggingTest, InvalidEnvValuesFallBackToDefaults) {
  setenv("INFERX_LOG_LEVEL", "loud", 1);
  setenv("INFERX_VLOG", "-2", 1);
  const auto config = LoggingConfigFromEnv();
  EXPECT_EQ(config.min_level, LogLevel::kInfo);
  EXPECT_EQ(config.vlog_verbosity, 0);
}

TEST(InitializeLogging, AcceptsValidConfigAndRejectsNegativeVlog) {
  EXPECT_TRUE(InitializeLogging({LogLevel::kError, 0}).ok());
  EXPECT_FALSE(InitializeLogging({LogLevel::kInfo, -1}).ok());
}

TEST(Logging, EmitsWithoutCrashing) {
  // Before or after initialization, sites must be safe to hit; suppression
  // is the sink's job, not the caller's.
  InitializeLogging({LogLevel::kInfo, 0});
  INFERX_LOG(WARNING) << "logging smoke test";
  INFERX_VLOG(1) << "vlog smoke test";
  INFERX_LOG(INFO) << absl::StrCat("concatenated ", 42);
  SUCCEED();
}

}  // namespace
}  // namespace inferx::core
