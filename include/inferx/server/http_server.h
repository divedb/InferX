// The coroutine HTTP front end for `inferx serve` (server.md §4-5): an
// accept loop and one coroutine per connection, built on Boost.Beast's
// awaitable support. This layer knows routing and framing only; API
// semantics live behind RequestDispatcher (implemented in serve.cc).
#ifndef INFERX_SERVER_HTTP_SERVER_H_
#define INFERX_SERVER_HTTP_SERVER_H_

#include <cstdint>
#include <string>

#include <boost/asio.hpp>
#include <boost/beast.hpp>

namespace inferx::server {

namespace net = boost::asio;
namespace beast = boost::beast;
namespace http = beast::http;

/// \brief Application hooks. `HandleAdmin` answers cheap synchronous
/// endpoints; `HandleCompletions` owns the full response for
/// POST /v1/completions, streaming included.
class RequestDispatcher {
 public:
  virtual ~RequestDispatcher() = default;

  /// \brief Handles a non-completions request. Always fills
  /// `status/content_type/body`; returns false to fall through to the
  /// built-in 404.
  virtual bool HandleAdmin(const std::string& method, const std::string& target,
                           int& status, std::string& content_type,
                           std::string& body) = 0;

  /// \brief Writes the complete response (headers, SSE stream or error) for
  /// a completions request onto `stream`. Runs inside the session
  /// coroutine; write failures must propagate as exceptions (the session
  /// treats them as a dead connection).
  virtual net::awaitable<void> HandleCompletions(
      beast::tcp_stream& stream, const http::request<http::string_body>& req) = 0;
};

class HttpServer {
 public:
  HttpServer(net::io_context& io, std::string host, std::uint16_t port,
             RequestDispatcher& dispatcher);

  /// \brief Starts accepting. Spawns the listener coroutine; failure to
  /// bind throws.
  void Listen();

  /// \brief Stops accepting new connections. Idempotent; in-flight sessions
  /// finish on their own.
  void Stop();

 private:
  net::awaitable<void> ListenLoop();
  net::awaitable<void> Session(beast::tcp_stream stream);

  net::io_context& io_;
  std::string host_;
  std::uint16_t port_;
  RequestDispatcher& dispatcher_;
  std::shared_ptr<net::ip::tcp::acceptor> acceptor_;
};

}  // namespace inferx::server

#endif  // INFERX_SERVER_HTTP_SERVER_H_
