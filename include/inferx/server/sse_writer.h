/// \brief Response writer for the HTTP sessions
/// (docs/inference_server_design.md §7.3/§9.1, reconciled to InferX).
///
/// One writer per HTTP exchange. Unary responses go through WriteJson;
/// streaming responses commit headers via BeginStream and then write SSE
/// frames directly — no unbounded queue exists, which is the doc's
/// backpressure contract: the caller cannot produce the next frame until
/// the previous write completes. All write methods return false on socket
/// failure (the caller must cancel engine work and stop writing) instead
/// of throwing, so handlers stay exception-free on the hot path.
#ifndef INFERX_SERVER_SSE_WRITER_H_
#define INFERX_SERVER_SSE_WRITER_H_

#include <chrono>
#include <cstdint>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <boost/asio.hpp>
#include <boost/beast.hpp>

#include "nlohmann/json.hpp"

namespace inferx {

namespace net = boost::asio;
namespace beast = boost::beast;
namespace http = beast::http;

class ResponseWriter {
 public:
  using Header = std::pair<std::string, std::string>;

  /// The stream must outlive the writer. Each write applies the write
  /// timeout to the underlying tcp_stream. `keep_alive` is declared on
  /// every response per the request's preference.
  ResponseWriter(beast::tcp_stream& stream, bool keep_alive,
                 std::chrono::milliseconds write_timeout = std::chrono::milliseconds(30000))
      : stream_(stream), keep_alive_(keep_alive), write_timeout_(write_timeout) {}

  ResponseWriter(const ResponseWriter&) = delete;
  ResponseWriter& operator=(const ResponseWriter&) = delete;

  /// \brief One complete JSON response. Exactly-once: the second call
  /// returns false without writing.
  net::awaitable<bool> WriteJson(int status, const nlohmann::json& body,
                                 std::vector<Header> extra_headers = {});

  /// \brief Empty-bodied status response (e.g. vLLM's /health).
  net::awaitable<bool> WriteEmpty(int status);

  /// \brief Commits SSE headers exactly once (200,
  /// text/event-stream; charset=utf-8, no-cache/no-transform,
  /// X-Accel-Buffering: no). After this the HTTP status is fixed.
  net::awaitable<bool> BeginStream();

  /// \brief Writes raw SSE bytes (frames are composed by the caller via
  /// the api.h helpers). One write per await; no coalescing buffer.
  net::awaitable<bool> WriteRaw(std::string_view bytes);

  /// \brief Writes the terminal `data: [DONE]\n\n`; idempotent per writer.
  net::awaitable<bool> WriteDone();

  bool committed() const { return committed_; }
  bool failed() const { return failed_; }

 private:
  net::awaitable<bool> WriteAll(std::string_view bytes);

  beast::tcp_stream& stream_;
  bool keep_alive_ = true;
  std::chrono::milliseconds write_timeout_;
  bool committed_ = false;
  bool failed_ = false;
  bool done_written_ = false;
};

}  // namespace inferx

#endif  // INFERX_SERVER_SSE_WRITER_H_
