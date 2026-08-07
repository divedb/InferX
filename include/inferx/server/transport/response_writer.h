#pragma once

#include <folly/CancellationToken.h>
#include <folly/coro/Task.h>

#include <boost/beast/http.hpp>
#include <string>

#include "inferx/core/status.h"

namespace inferx::server::transport {

namespace http = boost::beast::http;

using HttpRequest = http::request<http::string_body>;
using HttpResponse = http::response<http::string_body>;
using HttpStreamHead = http::response<http::empty_body>;

/// Inference-specific asynchronous response sink implemented by the Beast
/// transport adapter.
///
/// A writer is confined to one request. Calls are serialized: the next call is
/// legal only after the preceding task completes. Exactly one of WriteResponse
/// or StartStream starts a response, and a stream ends with Finish. Beast owns
/// HTTP semantics; this interface adds coroutine sequencing and streaming
/// backpressure without introducing a second HTTP representation.
class ResponseWriter {
 public:
  virtual ~ResponseWriter() = default;

  virtual folly::coro::Task<Status> WriteResponse(
      HttpResponse response, folly::CancellationToken cancellation) = 0;

  virtual folly::coro::Task<Status> StartStream(
      HttpStreamHead head, folly::CancellationToken cancellation) = 0;

  /// Writes already-framed bytes. SSE framing belongs to the protocol layer,
  /// while HTTP chunk framing belongs to the transport implementation.
  virtual folly::coro::Task<Status> Write(
      std::string data, folly::CancellationToken cancellation) = 0;

  virtual folly::coro::Task<Status> Finish(
      folly::CancellationToken cancellation) = 0;
};

/// Coroutine route handler. Beast types remain intact through routing; parsed
/// OpenAI domain objects are created by the protocol layer, not the transport.
class RequestHandler {
 public:
  virtual ~RequestHandler() = default;
  virtual folly::coro::Task<void> Handle(
      HttpRequest request, ResponseWriter& response,
      folly::CancellationToken cancellation) = 0;
};

}  // namespace inferx::server::transport
