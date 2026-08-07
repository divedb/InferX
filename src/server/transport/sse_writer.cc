#include "inferx/server/transport/sse_writer.h"

#include <boost/beast/http.hpp>

#include <utility>

namespace inferx::server::transport {

std::string SseWriter::Frame(std::string_view prefix, std::string_view value) {
  std::string output;
  size_t begin = 0;
  do {
    const size_t newline = value.find('\n', begin);
    output.append(prefix);
    output.append(value.substr(begin, newline - begin));
    output.push_back('\n');
    if (newline == std::string_view::npos) break;
    begin = newline + 1;
  } while (begin <= value.size());
  output.push_back('\n');
  return output;
}

folly::coro::Task<Status> SseWriter::Start(
    unsigned version, bool keep_alive, std::string request_id,
    folly::CancellationToken cancellation) {
  if (writer_ == nullptr) co_return FailedPreconditionError("SSE writer is null");
  HttpStreamHead head{boost::beast::http::status::ok, version};
  head.set(boost::beast::http::field::content_type, "text/event-stream");
  head.set(boost::beast::http::field::cache_control, "no-cache");
  head.set(boost::beast::http::field::connection, "keep-alive");
  head.set("X-Accel-Buffering", "no");
  if (!request_id.empty()) head.set("X-Request-ID", std::move(request_id));
  head.keep_alive(keep_alive);
  head.chunked(true);
  co_return co_await writer_->StartStream(std::move(head), cancellation);
}

folly::coro::Task<Status> SseWriter::Event(
    std::string data, folly::CancellationToken cancellation) {
  co_return co_await writer_->Write(Frame("data: ", data), cancellation);
}

folly::coro::Task<Status> SseWriter::Comment(
    std::string comment, folly::CancellationToken cancellation) {
  co_return co_await writer_->Write(Frame(": ", comment), cancellation);
}

folly::coro::Task<Status> SseWriter::Finish(
    folly::CancellationToken cancellation) {
  co_return co_await writer_->Finish(cancellation);
}

}  // namespace inferx::server::transport
