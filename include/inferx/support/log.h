#pragma once

#include <optional>
#include <ostream>
#include <string>
#include <string_view>

#include "absl/log/check.h"
#include "absl/log/log.h"

namespace inferx {

struct LogOptions {
  std::string min_level = "info";
  int verbosity = 0;
  bool json = false;
  std::optional<std::string> file;
};

// Initializes process-wide logging. Call once after command-line parsing and
// before starting worker threads. Logging before this call is still safe and
// is written to stderr by Abseil.
void InitLogging(const LogOptions& options = {});

// A deliberately tiny request-id formatter. Keeping the prefix uniform makes
// all text and JSON logs grep-able without introducing thread-local context.
class RequestLogPrefix {
 public:
  explicit RequestLogPrefix(std::string_view request_id)
      : request_id_(request_id) {}

  friend std::ostream& operator<<(std::ostream& out,
                                  const RequestLogPrefix& prefix) {
    return out << '[' << prefix.request_id_ << "] ";
  }

 private:
  std::string_view request_id_;
};

inline RequestLogPrefix Rid(std::string_view request_id) {
  return RequestLogPrefix(request_id);
}

}  // namespace inferx
