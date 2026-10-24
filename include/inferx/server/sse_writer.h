#ifndef INFERX_SERVER_SSE_WRITER_H_
#define INFERX_SERVER_SSE_WRITER_H_

#include <chrono>
#include <cstddef>
#include <string>
#include <string_view>
#include <vector>

#include <boost/asio/any_io_executor.hpp>
#include <boost/asio/awaitable.hpp>
#include <boost/asio/buffer.hpp>
#include <boost/beast/core/tcp_stream.hpp>

#include "inferx/server/http_types.h"
#include "inferx/server/status.h"

namespace inferx::server {

/// \brief Streaming response writer contract. After `Begin` succeeds the
///        HTTP status can no longer change; every method must be called on
///        the session strand.
class IStreamWriter {
 public:
  virtual ~IStreamWriter() = default;

  /// Sends status line and headers exactly once.
  virtual boost::asio::awaitable<Result<void>> Begin(int status,
                                                     HeaderMap headers) = 0;

  /// Writes one SSE data frame: "data: " + json + "\n\n". `json` must be
  /// valid UTF-8 and must not contain raw newlines outside JSON string
  /// escapes; the writer does not parse it.
  virtual boost::asio::awaitable<Result<void>> WriteEvent(
      std::string_view json) = 0;

  /// Writes an SSE comment, e.g. ": keep-alive\n\n".
  virtual boost::asio::awaitable<Result<void>> WriteComment(
      std::string_view comment) = 0;

  /// Writes the OpenAI terminal frame: "data: [DONE]\n\n".
  virtual boost::asio::awaitable<Result<void>> WriteDone() = 0;

  /// Best-effort terminal cleanup. Never throws; no writes after Abort.
  virtual void Abort() noexcept = 0;

  [[nodiscard]] virtual bool committed() const noexcept = 0;
  [[nodiscard]] virtual std::size_t bytes_written() const noexcept = 0;
};

/// \brief SSE framing over a Beast TCP stream using chunked transfer
///        coding (RFC 9112 §7.1): `make_chunk` frames each event and
///        `make_chunk_last` terminates the body. One coalesced write per
///        await: the caller cannot produce the next frame until the
///        current write completes, which is the backpressure contract.
class SseWriter final : public IStreamWriter {
 public:
  /// The stream must outlive the writer. All methods must be called on the
  /// strand owning `strand_executor`; `write_timeout` applies to each write.
  SseWriter(boost::asio::any_io_executor strand_executor,
            boost::beast::tcp_stream& stream,
            std::chrono::milliseconds write_timeout);

  boost::asio::awaitable<Result<void>> Begin(int status,
                                             HeaderMap headers) override;
  boost::asio::awaitable<Result<void>> WriteEvent(
      std::string_view json) override;
  boost::asio::awaitable<Result<void>> WriteComment(
      std::string_view comment) override;
  /// Writes the OpenAI terminal frame and the zero-length last chunk in one
  /// write; the stream is finished afterwards.
  boost::asio::awaitable<Result<void>> WriteDone() override;
  void Abort() noexcept override;

  [[nodiscard]] bool committed() const noexcept override;
  [[nodiscard]] std::size_t bytes_written() const noexcept override;

 private:
  boost::asio::awaitable<Result<void>> WriteAll(std::string bytes);
  boost::asio::awaitable<Result<void>> WriteBuffers(
      std::vector<boost::asio::const_buffer> buffers);
  boost::asio::any_io_executor strand_;
  boost::beast::tcp_stream& stream_;
  std::chrono::milliseconds write_timeout_;
  bool committed_ = false;
  bool finished_ = false;
  bool aborted_ = false;
  std::size_t bytes_written_ = 0;
};

}  // namespace inferx::server

#endif  // INFERX_SERVER_SSE_WRITER_H_
