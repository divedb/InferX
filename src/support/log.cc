#include "inferx/support/log.h"

#include <cstdio>
#include <fstream>
#include <memory>
#include <mutex>
#include <string>

#include "absl/base/log_severity.h"
#include "absl/log/globals.h"
#include "absl/log/initialize.h"
#include "absl/log/log_entry.h"
#include "absl/log/log_sink.h"
#include "absl/log/log_sink_registry.h"
#include "absl/strings/str_cat.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "inferx/support/json.h"

namespace inferx {
namespace {

absl::LogSeverityAtLeast ParseLevel(const std::string& level) {
  if (level == "warning") return absl::LogSeverityAtLeast::kWarning;
  if (level == "error") return absl::LogSeverityAtLeast::kError;
  return absl::LogSeverityAtLeast::kInfo;
}

std::string SeverityName(absl::LogSeverity severity) {
  switch (severity) {
    case absl::LogSeverity::kInfo: return "INFO";
    case absl::LogSeverity::kWarning: return "WARNING";
    case absl::LogSeverity::kError: return "ERROR";
    case absl::LogSeverity::kFatal: return "FATAL";
  }
  return "UNKNOWN";
}

std::string JsonLine(const absl::LogEntry& entry) {
  std::string out = "{\"ts\":";
  AppendJsonString(absl::FormatTime("%Y-%m-%dT%H:%M:%E6SZ", entry.timestamp(),
                                    absl::UTCTimeZone()), &out);
  out += ",\"severity\":";
  AppendJsonString(SeverityName(entry.log_severity()), &out);
  out += ",\"file\":";
  AppendJsonString(entry.source_basename(), &out);
  absl::StrAppend(&out, ",\"line\":", entry.source_line(),
                  ",\"tid\":", entry.tid(), ",\"msg\":");
  AppendJsonString(entry.text_message(), &out);
  out += "}\n";
  return out;
}

class InferxLogSink final : public absl::LogSink {
 public:
  InferxLogSink(bool json_stderr, const std::optional<std::string>& path)
      : json_stderr_(json_stderr) {
    if (path.has_value()) {
      file_.open(*path, std::ios::out | std::ios::app);
      if (!file_) {
        std::fprintf(stderr, "failed to open log file %s\n", path->c_str());
      }
    }
  }

  void Send(const absl::LogEntry& entry) override {
    const std::string json = JsonLine(entry);
    std::lock_guard<std::mutex> lock(mutex_);
    if (json_stderr_) {
      std::fwrite(json.data(), 1, json.size(), stderr);
    }
    if (file_) {
      if (json_stderr_) {
        file_ << json;
      } else {
        file_ << entry.text_message_with_prefix_and_newline();
      }
      file_.flush();
    }
  }

  void Flush() override {
    std::lock_guard<std::mutex> lock(mutex_);
    if (json_stderr_) std::fflush(stderr);
    if (file_) file_.flush();
  }

 private:
  bool json_stderr_;
  std::ofstream file_;
  std::mutex mutex_;
};

std::mutex g_init_mutex;
std::unique_ptr<InferxLogSink> g_sink;
bool g_initialized = false;

}  // namespace

void InitLogging(const LogOptions& options) {
  std::lock_guard<std::mutex> lock(g_init_mutex);
  if (!g_initialized) {
    absl::InitializeLog();
    g_initialized = true;
  }
  if (g_sink) {
    absl::RemoveLogSink(g_sink.get());
    g_sink.reset();
  }

  const absl::LogSeverityAtLeast level = ParseLevel(options.min_level);
  absl::SetMinLogLevel(level);
  absl::SetGlobalVLogLevel(options.verbosity);
  absl::SetStderrThreshold(options.json
                               ? absl::LogSeverityAtLeast::kInfinity
                               : level);

  if (options.json || options.file.has_value()) {
    g_sink = std::make_unique<InferxLogSink>(options.json, options.file);
    absl::AddLogSink(g_sink.get());
  }
}

}  // namespace inferx
