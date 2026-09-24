/// \file
/// \brief Logging facade over Abseil's logging library.
///
/// One include, one spelling, engine-wide: leveled `INFERX_LOG` for
/// operational messages, verbosity-gated `INFERX_VLOG` for milestone and
/// hot-path tracing. Abseil defaults (INFO and up to stderr) apply until
/// `InitializeLogging` runs, so early startup and tests log sanely with no
/// setup. Statuses remain the error channel; logs are for humans.

#ifndef INFERX_CORE_LOGGING_H_
#define INFERX_CORE_LOGGING_H_

#include <string_view>

#include "absl/log/absl_log.h"
#include "inferx/core/status.h"

/// \brief Leveled logging; streams like Abseil's:
///        `INFERX_LOG(INFO) << "mapped " << shards << " shard(s)";`
/// Severities: INFO, WARNING, ERROR, FATAL (terminates the process).
#define INFERX_LOG(severity) ABSL_LOG(severity)

/// \brief Verbosity-gated logging for milestones and hot-path tracing:
///        `INFERX_VLOG(2) << "step: tokens=" << n;`
/// Threshold comes from the INFERX_VLOG environment level via
/// InitializeLogging; a disabled site costs one global load, so VLOGs may
/// live on per-step paths.
#define INFERX_VLOG(verbosity) ABSL_VLOG(verbosity)

namespace inferx::core {

/// \brief Minimum severity that reaches the log.
enum class LogLevel {
  kInfo = 0,
  kWarning = 1,
  kError = 2,
};

/// \brief Parses "info" | "warning" | "error" (lowercase) into a level.
StatusOr<LogLevel> ParseLogLevel(std::string_view name);

/// \brief Logging thresholds.
struct LoggingConfig {
  LogLevel min_level = LogLevel::kInfo;  ///< Severities below this drop.
  int vlog_verbosity = 0;                ///< INFERX_VLOG threshold; 0 silences all.
};

/// \brief Reads INFERX_LOG_LEVEL and INFERX_VLOG, falling back to defaults
///        (with a stderr note) on unparsable values.
LoggingConfig LoggingConfigFromEnv();

/// \brief Configures the global Abseil log from `config`.
///
/// Call once at process start, before worker threads spawn. The returned
/// Status only rejects direct misuse (negative verbosity); the
/// environment-derived config is always valid.
Status InitializeLogging(const LoggingConfig& config);

}  // namespace inferx::core

#endif  // INFERX_CORE_LOGGING_H_
