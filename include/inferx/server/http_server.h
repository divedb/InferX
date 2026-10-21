// The coroutine HTTP/1.1 front end for `inferx serve`
// (docs/inference_server_design.md §5-6, reconciled to InferX): an accept
// loop and one session coroutine per connection over Boost.Beast. This
// layer owns framing, limits, keep-alive, drain state, and the standard
// error envelopes (404/405/413); all API semantics live behind
// RequestHandler (implemented by openai_handlers.cc).
#ifndef INFERX_SERVER_HTTP_SERVER_H_
#define INFERX_SERVER_HTTP_SERVER_H_

#include <atomic>
#include <chrono>
#include <cstdint>
#include <memory>
#include <string>

#include <boost/asio.hpp>
#include <boost/beast.hpp>

#include "inferx/server/sse_writer.h"

namespace inferx {

namespace net = boost::asio;
namespace beast = boost::beast;
namespace http = beast::http;

/// \brief One HTTP exchange handed to the handler. The handler must write
/// exactly one response (unary via writer.WriteJson, streaming via
/// BeginStream + raw frames); write failures surface as `false` returns
/// from the writer so the handler can cancel engine work.
struct HttpExchange {
  http::request<http::string_body> request;
  std::string method;
  std::string path;    ///< Target without the query string.
  std::string target;  ///< Full request target.
  ResponseWriter writer;
};

class RequestHandler {
 public:
  virtual ~RequestHandler() = default;

  /// \brief Handles one request. Returns true when a route matched (the
  /// response has been written); false makes the session emit a 404
  /// envelope. Method mismatches are the handler's 405 to emit.
  virtual net::awaitable<bool> Handle(HttpExchange& exchange) = 0;
};

class HttpServer {
 public:
  HttpServer(net::io_context& io, std::string host, std::uint16_t port,
             RequestHandler& handler);

  /// \brief Binds and starts accepting; returns the bound port (useful
  /// when port 0 is requested). Throws on bind failure.
  std::uint16_t Listen();

  /// \brief Stops accepting new connections. Idempotent; in-flight
  /// sessions finish on their own.
  void Stop();

  /// \brief Latches drain mode: new non-health requests get 503 with
  /// `Connection: close`. Idempotent; thread-safe.
  void SetDraining(bool draining);

  bool draining() const { return draining_->load(); }

 private:
  net::awaitable<void> ListenLoop();
  net::awaitable<void> Session(beast::tcp_stream stream);

  net::io_context& io_;
  std::string host_;
  std::uint16_t port_;
  RequestHandler& handler_;
  std::shared_ptr<net::ip::tcp::acceptor> acceptor_;
  std::shared_ptr<std::atomic<bool>> draining_ = std::make_shared<std::atomic<bool>>(false);
};

}  // namespace inferx

#endif  // INFERX_SERVER_HTTP_SERVER_H_
