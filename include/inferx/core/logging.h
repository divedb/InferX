#ifndef INFERX_CORE_LOGGING_H_
#define INFERX_CORE_LOGGING_H_

#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/log/absl_check.h"
#include "absl/log/absl_log.h"
#include "inferx/core/status.h"

/// \brief Leveled logging; streams like Abseil's:
///        `INFERX_LOG(INFO) << "mapped " << shards << " shard(s)";`
/// Severities: INFO, WARNING, ERROR, FATAL. FATAL always reaches the log
/// (regardless of LoggingConfig::min_level) and terminates the process.
#define INFERX_LOG(severity) ABSL_LOG(severity)

/// \brief Verbosity-gated logging for milestones and hot-path tracing:
///        `INFERX_VLOG(2) << "step: tokens=" << n;`
/// Emitted when the effective verbosity (global, or a matching per-module
/// override) is >= the given level. A disabled site is cheap, so VLOGs may
/// live on per-step paths.
#define INFERX_VLOG(verbosity) ABSL_VLOG(verbosity)

/// \brief Invariant checks; abort with a message when `condition` is false.
///        INFERX_DCHECK compiles away in release (NDEBUG) builds.
///        `INFERX_CHECK(shard != nullptr) << "missing shard " << index;`
#define INFERX_CHECK(condition) ABSL_CHECK(condition)
#define INFERX_DCHECK(condition) ABSL_DCHECK(condition)

namespace inferx::core {

/// \brief Minimum severity that reaches the log. FATAL is not listed
///        because it is never filtered.
enum class LogLevel {
  kInfo = 0,
  kWarning = 1,
  kError = 2,
};

/// \brief Parses "info" | "warning" | "error" (lowercase) into a level.
///        Used by the environment/file loader and by CLI flag handling.
StatusOr<LogLevel> ParseLogLevel(std::string_view name);

/// \brief Logging thresholds and sinks.
struct LoggingConfig {
  /// Severities below this are dropped.
  LogLevel min_level = LogLevel::kInfo;

  /// Global INFERX_VLOG threshold; 0 silences all VLOG sites that have no
  /// module override. Must be >= 0.
  int vlog_verbosity = 0;

  /// Per-module VLOG overrides as {module pattern, verbosity}, e.g.
  /// {"model_runner", 2}. Lets one component be traced without flooding
  /// the rest. Verbosities must be >= 0.
  std::vector<std::pair<std::string, int>> vlog_modules;

  /// Additional log file written alongside stderr; empty means stderr only.
  std::string log_file;
};

/// \brief Builds the startup config from the environment.
///
/// Layers, lowest to highest precedence:
///   1. LoggingConfig defaults.
///   2. The `logging:` section of the YAML file named by INFERX_CONFIG_FILE
///      (keys: level, vlog, vlog_modules, file). Unset means no file.
///   3. INFERX_LOG_LEVEL (info | warning | error) and INFERX_VLOG (integer).
///
/// CLI flags are applied by the caller on top of the returned value.
///
/// Fails, rather than falling back to defaults, when INFERX_CONFIG_FILE
/// points at a missing or unparsable file, when the `logging:` section has
/// unknown keys, or when an INFERX_* value cannot be parsed. Logging is not
/// initialized yet, so the caller reports the Status itself (stderr) and
/// exits.
StatusOr<LoggingConfig> LoggingConfigFromEnvironment();

/// \brief Installs the Abseil log sinks and applies `config`.
///
/// Call exactly once, at the start of main after config and CLI parsing and
/// before worker threads spawn. A second call returns an error Status
/// instead of reinstalling sinks. Also rejects negative verbosities and an
/// unopenable `log_file`.
Status InitializeLogging(const LoggingConfig& config);

}  // namespace inferx::core

#endif  // INFERX_CORE_LOGGING_H_