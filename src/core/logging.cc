#include "inferx/core/logging.h"

#include <charconv>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <optional>
#include <string>
#include <system_error>

#include "absl/log/globals.h"
#include "absl/log/initialize.h"

namespace inferx::core {
namespace {

absl::LogSeverityAtLeast Severity(LogLevel level) {
  switch (level) {
    case LogLevel::kInfo:
      return absl::LogSeverityAtLeast::kInfo;
    case LogLevel::kWarning:
      return absl::LogSeverityAtLeast::kWarning;
    case LogLevel::kError:
      return absl::LogSeverityAtLeast::kError;
  }
  return absl::LogSeverityAtLeast::kError;
}

std::optional<std::string> Env(const char* name) {
  const char* value = std::getenv(name);
  if (value == nullptr) return std::nullopt;
  return std::string(value);
}

void NoteInvalid(const char* variable, const std::string& value) {
  std::fprintf(stderr, "inferx: invalid %s '%s'; using the default\n", variable,
               value.c_str());
}

}  // namespace

StatusOr<LogLevel> ParseLogLevel(std::string_view name) {
  if (name == "info") return LogLevel::kInfo;
  if (name == "warning") return LogLevel::kWarning;
  if (name == "error") return LogLevel::kError;
  return InvalidArgumentError("unknown log level '", name,
                              "' (want info, warning, or error)");
}

LoggingConfig LoggingConfigFromEnv() {
  LoggingConfig config;
  if (auto level = Env("INFERX_LOG_LEVEL")) {
    auto parsed = ParseLogLevel(*level);
    if (parsed.ok()) {
      config.min_level = *parsed;
    } else {
      NoteInvalid("INFERX_LOG_LEVEL", *level);
    }
  }
  if (auto vlog = Env("INFERX_VLOG")) {
    int value = 0;
    const char* const end = vlog->data() + vlog->size();
    const auto [ptr, ec] = std::from_chars(vlog->data(), end, value);
    if (ec == std::errc() && ptr == end && value >= 0) {
      config.vlog_verbosity = value;
    } else {
      NoteInvalid("INFERX_VLOG", *vlog);
    }
  }
  return config;
}

Status InitializeLogging(const LoggingConfig& config) {
  if (config.vlog_verbosity < 0) {
    return InvalidArgumentError("vlog verbosity must be non-negative, got ",
                                config.vlog_verbosity);
  }
  // absl::InitializeLog() aborts on a second call, so guard it: repeated
  // InitializeLogging (tests, embedding apps) only re-applies thresholds.
  static std::once_flag init_once;
  std::call_once(init_once, [] { absl::InitializeLog(); });
  // Abseil's post-initialization stderr threshold defaults to ERROR; without
  // registered sinks that would drop INFO and WARNING entirely. Route the
  // minimum level to stderr so "initialize then log INFO" behaves like the
  // pre-initialization default.
  absl::SetStderrThreshold(Severity(config.min_level));
  absl::SetMinLogLevel(Severity(config.min_level));
  absl::SetGlobalVLogLevel(config.vlog_verbosity);
  return OkStatus();
}

}  // namespace inferx::core
