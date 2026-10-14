#include "inferx/core/logging.h"

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/log/globals.h"
#include "absl/log/initialize.h"
#include "absl/log/log_entry.h"
#include "absl/log/log_sink.h"
#include "absl/log/log_sink_registry.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "yaml-cpp/yaml.h"

namespace inferx::core {
namespace {

constexpr char kConfigFileEnv[] = "INFERX_CONFIG_FILE";
constexpr char kLogLevelEnv[] = "INFERX_LOG_LEVEL";
constexpr char kVlogEnv[] = "INFERX_VLOG";
constexpr char kLoggingSection[] = "logging";

// -- Small helpers ----------------------------------------------------------

/// Returns the variable's value, treating unset and empty the same way.
std::optional<std::string> GetEnv(const char* name) {
  const char* value = std::getenv(name);
  if (value == nullptr || *value == '\0') return std::nullopt;
  return std::string(value);
}

/// Prefixes an error with where the bad value came from.
Status WithContext(const Status& status, std::string_view context) {
  return InvalidArgumentError(absl::StrCat(context, ": ", status.message()));
}

StatusOr<int> ParseVerbosity(std::string_view text) {
  int value = 0;
  if (!absl::SimpleAtoi(text, &value) || value < 0) {
    return InvalidArgumentError(
        absl::StrCat("invalid verbosity '", text, "' (expected a non-negative integer)"));
  }
  return value;
}

// -- YAML overlay -----------------------------------------------------------

/// Applies one key of the `logging:` section to `config`.
Status ApplyYamlKey(const std::string& key, const YAML::Node& value, LoggingConfig* config) {
  if (key == "level") {
    StatusOr<absl::LogSeverityAtLeast> level = ParseLogLevel(value.Scalar());
    if (!level.ok()) return level.status();
    config->min_level = *level;
    return OkStatus();
  }
  if (key == "vlog") {
    StatusOr<int> verbosity = ParseVerbosity(value.Scalar());
    if (!verbosity.ok()) return verbosity.status();
    config->vlog_verbosity = *verbosity;
    return OkStatus();
  }
  if (key == "vlog_modules") {
    if (!value.IsMap()) {
      return InvalidArgumentError("expected a map of module pattern to verbosity");
    }
    std::vector<std::pair<std::string, int>> modules;
    for (const auto& module : value) {
      const std::string pattern = module.first.Scalar();
      if (pattern.empty()) return InvalidArgumentError("empty module pattern");
      StatusOr<int> verbosity = ParseVerbosity(module.second.Scalar());
      if (!verbosity.ok()) return WithContext(verbosity.status(), pattern);
      modules.emplace_back(pattern, *verbosity);
    }
    config->vlog_modules = std::move(modules);
    return OkStatus();
  }
  if (key == "file") {
    if (!value.IsScalar()) return InvalidArgumentError("expected a file path");
    config->log_file = value.Scalar();
    return OkStatus();
  }
  return InvalidArgumentError("unknown key (expected level, vlog, vlog_modules, or file)");
}

/// Overlays the `logging:` section of the YAML file at `path` onto `config`.
/// Other top-level sections (serve:, bench:, ...) belong to other components
/// and are ignored here.
Status ApplyYamlFile(const std::string& path, LoggingConfig* config) {
  try {
    const YAML::Node root = YAML::LoadFile(path);
    if (root.IsNull()) return OkStatus();  // Empty file.
    if (!root.IsMap()) {
      return InvalidArgumentError(absl::StrCat(path, ": top level must be a map"));
    }
    const YAML::Node section = root[kLoggingSection];
    if (!section || section.IsNull()) return OkStatus();
    if (!section.IsMap()) {
      return InvalidArgumentError(
          absl::StrCat(path, ": '", kLoggingSection, "' must be a map"));
    }
    for (const auto& entry : section) {
      const std::string key = entry.first.Scalar();
      Status status = ApplyYamlKey(key, entry.second, config);
      if (!status.ok()) {
        return WithContext(status, absl::StrCat(path, ": ", kLoggingSection, ".", key));
      }
    }
    return OkStatus();
  } catch (const YAML::BadFile&) {
    return NotFoundError(
        absl::StrCat(kConfigFileEnv, " points at '", path, "', which cannot be read"));
  } catch (const YAML::Exception& e) {
    return InvalidArgumentError(absl::StrCat("cannot parse '", path, "': ", e.what()));
  }
}

// -- File sink --------------------------------------------------------------

/// Appends every log entry to a file. Lives for the whole process: it is
/// deliberately never removed or destroyed so that logging during static
/// destruction cannot touch a dead sink. stdio serializes concurrent writes.
class FileLogSink final : public absl::LogSink {
 public:
  explicit FileLogSink(std::FILE* file) : file_(file) {}

  void Send(const absl::LogEntry& entry) override {
    const absl::string_view text = entry.text_message_with_prefix_and_newline();
    std::fwrite(text.data(), 1, text.size(), file_);
    if (entry.log_severity() >= absl::LogSeverity::kWarning) std::fflush(file_);
  }

  void Flush() override { std::fflush(file_); }

 private:
  std::FILE* file_;
};

Status ValidateConfig(const LoggingConfig& config) {
  if (config.min_level > absl::LogSeverityAtLeast::kError) {
    return InvalidArgumentError("min_level must be kInfo, kWarning, or kError");
  }
  if (config.vlog_verbosity < 0) {
    return InvalidArgumentError("vlog_verbosity must be >= 0");
  }
  for (const auto& [pattern, verbosity] : config.vlog_modules) {
    if (pattern.empty()) return InvalidArgumentError("vlog_modules: empty module pattern");
    if (verbosity < 0) {
      return InvalidArgumentError(absl::StrCat("vlog_modules.", pattern, " must be >= 0"));
    }
  }
  return OkStatus();
}

std::mutex g_init_mutex;
bool g_initialized = false;

}  // namespace

StatusOr<absl::LogSeverityAtLeast> ParseLogLevel(std::string_view name) {
  if (name == "info") return absl::LogSeverityAtLeast::kInfo;
  if (name == "warning") return absl::LogSeverityAtLeast::kWarning;
  if (name == "error") return absl::LogSeverityAtLeast::kError;
  return InvalidArgumentError(
      absl::StrCat("invalid log level '", name, "' (expected info, warning, or error)"));
}

StatusOr<LoggingConfig> LoggingConfigFromEnvironment() {
  LoggingConfig config;

  if (const std::optional<std::string> path = GetEnv(kConfigFileEnv)) {
    Status status = ApplyYamlFile(*path, &config);
    if (!status.ok()) return status;
  }

  if (const std::optional<std::string> value = GetEnv(kLogLevelEnv)) {
    StatusOr<absl::LogSeverityAtLeast> level = ParseLogLevel(*value);
    if (!level.ok()) return WithContext(level.status(), kLogLevelEnv);
    config.min_level = *level;
  }

  if (const std::optional<std::string> value = GetEnv(kVlogEnv)) {
    StatusOr<int> verbosity = ParseVerbosity(*value);
    if (!verbosity.ok()) return WithContext(verbosity.status(), kVlogEnv);
    config.vlog_verbosity = *verbosity;
  }

  return config;
}

Status InitializeLogging(const LoggingConfig& config) {
  std::lock_guard<std::mutex> lock(g_init_mutex);
  if (g_initialized) {
    return FailedPreconditionError("InitializeLogging was already called");
  }

  // Everything that can fail happens before any global state changes, so a
  // rejected call leaves logging untouched.
  Status valid = ValidateConfig(config);
  if (!valid.ok()) return valid;

  std::FILE* log_file = nullptr;
  if (!config.log_file.empty()) {
    log_file = std::fopen(config.log_file.c_str(), "a");
    if (log_file == nullptr) {
      return InvalidArgumentError(
          absl::StrCat("cannot open log file '", config.log_file, "': ", std::strerror(errno)));
    }
  }

  absl::InitializeLog();

  // Abseil's stderr threshold defaults to ERROR, so it is set explicitly to
  // make min_level mean "what you see". VLOG messages are emitted at INFO
  // severity, so min_level above info also hides them.
  absl::SetMinLogLevel(config.min_level);
  absl::SetStderrThreshold(config.min_level);

  absl::SetGlobalVLogLevel(config.vlog_verbosity);
  for (const auto& [pattern, verbosity] : config.vlog_modules) {
    absl::SetVLogLevel(pattern, verbosity);
  }

  if (log_file != nullptr) {
    absl::AddLogSink(new FileLogSink(log_file));  // Intentionally leaked.
  }

  g_initialized = true;
  return OkStatus();
}

}  // namespace inferx::core