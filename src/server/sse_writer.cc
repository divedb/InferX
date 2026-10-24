#include "inferx/server/sse_writer.h"

#include <string>
#include <utility>
#include <vector>

#include <boost/asio/buffer.hpp>
#include <boost/asio/error.hpp>
#include <boost/asio/redirect_error.hpp>
#include <boost/asio/use_awaitable.hpp>
#include <boost/asio/write.hpp>
#include <boost/system/errc.hpp>
#include <boost/beast/http/chunk_encode.hpp>
#include <boost/beast/http/empty_body.hpp>
#include <boost/beast/http/message.hpp>
#include <boost/beast/http/write.hpp>

#include "inferx/server/api_error_mapper.h"
#include "inferx/server/json_codec.h"

namespace inferx::server {

namespace {

bool IsValidHeaderName(std::string_view name) {
  if (name.empty()) return false;
  for (const char c : name) {
    const bool token = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
                       (c >= '0' && c <= '9') || c == '-' || c == '_' ||
                       c == '!' || c == '#' || c == '$' || c == '%' ||
                       c == '&' || c == '\'' || c == '*' || c == '+' ||
                       c == '.' || c == '^' || c == '`' || c == '~' || c == '|';
    if (!token) return false;
  }
  return true;
}

bool IsValidHeaderValue(std::string_view value) {
  for (const char c : value) {
    const auto byte = static_cast<unsigned char>(c);
    if (byte < 0x20 && byte != '\t') return false;
    if (byte == 0x7F) return false;
  }
  return true;
}

/// Flattens a Beast chunk buffer sequence (size line, CRLFs, payload) into
/// the const_buffer list WriteBuffers takes.
template <class ChunkSequence>
void AppendChunk(std::vector<boost::asio::const_buffer>& out,
                 const ChunkSequence& chunk) {
  out.insert(out.end(), chunk.begin(), chunk.end());
}

ApiError TransportError(const boost::system::error_code& ec) {
  const bool disconnect =
      ec == boost::system::errc::connection_reset ||
      ec == boost::system::errc::broken_pipe ||
      ec == boost::asio::error::connection_aborted ||
      ec == boost::asio::error::eof ||
      ec == boost::asio::error::operation_aborted;
  return ApiError{disconnect ? 499 : 500,
                  disconnect ? ApiErrorKind::kEngine : ApiErrorKind::kInternal,
                  disconnect ? "client_disconnected" : "internal_error",
                  disconnect ? "client_disconnected" : "write_failed",
                  "stream write failed", std::nullopt, std::nullopt, false, {}};
}

}  // namespace

SseWriter::SseWriter(boost::asio::any_io_executor strand_executor,
                     boost::beast::tcp_stream& stream,
                     std::chrono::milliseconds write_timeout)
    : strand_(std::move(strand_executor)),
      stream_(stream),
      write_timeout_(write_timeout) {}

boost::asio::awaitable<Result<void>> SseWriter::Begin(int status,
                                                      HeaderMap headers) {
  if (committed_) {
    co_return ApiErrorMapper::Internal("response_committed",
                                       "response already started");
  }
  for (const auto& field : headers) {
    if (!IsValidHeaderName(field.name_string()) ||
        !IsValidHeaderValue(field.value())) {
      co_return ApiErrorMapper::Internal("invalid_response_header",
                                         "invalid response header");
    }
  }
  boost::beast::http::response<boost::beast::http::empty_body> res{
      static_cast<boost::beast::http::status>(status), 11};
  // header<false, Fields> derives from Fields; assign through the base.
  static_cast<boost::beast::http::fields&>(res.base()) = std::move(headers);
  // Chunked framing keeps the persistent connection valid and lets
  // HTTP/1.1 proxies stream instead of buffering; the serializer emits
  // Transfer-Encoding from the chunked flag.
  res.chunked(true);
  res.keep_alive(true);

  boost::beast::http::response_serializer<boost::beast::http::empty_body>
      serializer{res};
  boost::system::error_code ec;
  stream_.expires_after(write_timeout_);
  co_await boost::beast::http::async_write_header(
      stream_, serializer,
      boost::asio::redirect_error(boost::asio::use_awaitable, ec));
  if (ec) {
    aborted_ = true;
    co_return TransportError(ec);
  }
  committed_ = true;
  co_return Result<void>{};
}

boost::asio::awaitable<Result<void>> SseWriter::WriteEvent(
    std::string_view json) {
  if (!committed_ || aborted_ || finished_) {
    co_return ApiErrorMapper::Internal("stream_not_writable",
                                       "Begin() was not called or the stream "
                                       "already finished or aborted");
  }
  if (!IsValidUtf8(json)) {
    co_return ApiErrorMapper::Internal("invalid_utf8",
                                       "engine emitted invalid UTF-8");
  }
  std::string frame;
  frame.reserve(json.size() + 8);
  frame.append("data: ");
  frame.append(json);
  frame.append("\n\n");
  // One coalesced chunk per await. No unbounded queue exists here: the
  // caller cannot produce the next frame until this write completes, which
  // is the backpressure contract.
  co_return co_await WriteAll(std::move(frame));
}

boost::asio::awaitable<Result<void>> SseWriter::WriteComment(
    std::string_view comment) {
  if (!committed_ || aborted_ || finished_) {
    co_return ApiErrorMapper::Internal("stream_not_writable",
                                       "Begin() was not called or the stream "
                                       "already finished or aborted");
  }
  std::string frame;
  frame.reserve(comment.size() + 4);
  frame.append(": ");
  frame.append(comment);
  frame.append("\n\n");
  co_return co_await WriteAll(std::move(frame));
}

boost::asio::awaitable<Result<void>> SseWriter::WriteDone() {
  if (!committed_ || aborted_ || finished_) {
    co_return ApiErrorMapper::Internal("stream_not_writable",
                                       "Begin() was not called or the stream "
                                       "already finished or aborted");
  }
  // One write carries the [DONE] chunk and the terminal chunk — a slow peer
  // sees the stream end atomically.
  static constexpr std::string_view kDone = "data: [DONE]\n\n";
  std::vector<boost::asio::const_buffer> buffers;
  AppendChunk(buffers, boost::beast::http::make_chunk(
                           boost::asio::const_buffer(kDone.data(), kDone.size())));
  AppendChunk(buffers, boost::beast::http::make_chunk_last());
  const auto result = co_await WriteBuffers(std::move(buffers));
  if (result.ok()) finished_ = true;
  co_return result;
}

void SseWriter::Abort() noexcept {
  if (aborted_) return;
  aborted_ = true;
  try {
    stream_.cancel();
  } catch (...) {
    // Best-effort cleanup only; the socket close is handled by the session.
  }
}

bool SseWriter::committed() const noexcept { return committed_; }
std::size_t SseWriter::bytes_written() const noexcept { return bytes_written_; }

boost::asio::awaitable<Result<void>> SseWriter::WriteAll(std::string bytes) {
  std::vector<boost::asio::const_buffer> buffers;
  AppendChunk(buffers, boost::beast::http::make_chunk(
                           boost::asio::const_buffer(bytes.data(), bytes.size())));
  // `bytes` outlives the co_await below: the coroutine frame holds it until
  // WriteBuffers completes.
  const auto result = co_await WriteBuffers(std::move(buffers));
  co_return result;
}

boost::asio::awaitable<Result<void>> SseWriter::WriteBuffers(
    std::vector<boost::asio::const_buffer> buffers) {
  boost::system::error_code ec;
  stream_.expires_after(write_timeout_);
  co_await boost::asio::async_write(
      stream_, buffers,
      boost::asio::redirect_error(boost::asio::use_awaitable, ec));
  if (ec) {
    aborted_ = true;
    co_return TransportError(ec);
  }
  for (const auto& buffer : buffers) bytes_written_ += buffer.size();
  co_return Result<void>{};
}

}  // namespace inferx::server
