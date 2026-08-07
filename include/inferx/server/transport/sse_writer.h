#pragma once

#include <folly/CancellationToken.h>
#include <folly/coro/Task.h>

#include <string>
#include <string_view>

#include "inferx/core/status.h"
#include "inferx/server/transport/response_writer.h"

namespace inferx::server::transport {

class SseWriter {
 public:
  explicit SseWriter(ResponseWriter* writer) : writer_(writer) {}

  folly::coro::Task<Status> Start(
      unsigned version, bool keep_alive, std::string request_id,
      folly::CancellationToken cancellation);
  folly::coro::Task<Status> Event(
      std::string data, folly::CancellationToken cancellation);
  folly::coro::Task<Status> Comment(
      std::string comment, folly::CancellationToken cancellation);
  folly::coro::Task<Status> Finish(
      folly::CancellationToken cancellation);

 private:
  static std::string Frame(std::string_view prefix, std::string_view value);
  ResponseWriter* writer_;
};

}  // namespace inferx::server::transport
