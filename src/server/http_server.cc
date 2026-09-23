// HttpServer implementation: listener + per-connection session coroutines.
// Conventions: every response declares keep-alive per the request; errors
// on the socket end the session coroutine quietly (disconnects are
// routine, not exceptional); detached coroutines never let exceptions
// escape (that would terminate the process).
#include "inferx/server/http_server.h"

#include <cstdio>
#include <memory>
#include <utility>

#include "inferx/server/api.h"

namespace inferx::server {
namespace {

// A request head must arrive within this window; generation itself is not
// bounded (expires_never during completions).
constexpr auto kReadTimeout = std::chrono::seconds(30);
// Requests larger than this are rejected before parsing.
constexpr std::size_t kMaxBodyBytes = 8u << 20;

http::response<http::string_body> MakeResponse(
    const http::request<http::string_body>& req, http::status status,
    std::string content_type, std::string body) {
  http::response<http::string_body> res{status, req.version()};
  res.set(http::field::server, "inferx");
  res.set(http::field::content_type, std::move(content_type));
  res.keep_alive(req.keep_alive());
  res.body() = std::move(body);
  res.prepare_payload();
  return res;
}

}  // namespace

HttpServer::HttpServer(net::io_context& io, std::string host, std::uint16_t port,
                       RequestDispatcher& dispatcher)
    : io_(io), host_(std::move(host)), port_(port), dispatcher_(dispatcher) {}

void HttpServer::Listen() {
  auto acceptor = std::make_shared<net::ip::tcp::acceptor>(io_.get_executor());
  net::ip::tcp::endpoint endpoint(net::ip::make_address(host_), port_);
  acceptor->open(endpoint.protocol());
  acceptor->set_option(net::socket_base::reuse_address(true));
  acceptor->bind(endpoint);
  acceptor->listen(net::socket_base::max_listen_connections);
  acceptor_ = acceptor;
  net::co_spawn(io_, ListenLoop(), net::detached);
}

void HttpServer::Stop() {
  if (!acceptor_) return;
  net::post(io_, [acceptor = acceptor_] {
    if (acceptor->is_open()) {
      boost::system::error_code ignored;
      acceptor->close(ignored);
    }
  });
}

net::awaitable<void> HttpServer::ListenLoop() {
  try {
    for (;;) {
      net::ip::tcp::socket socket = co_await acceptor_->async_accept(net::use_awaitable);
      if (!acceptor_->is_open()) break;
      net::co_spawn(io_, Session(beast::tcp_stream(std::move(socket))), net::detached);
    }
  } catch (const std::exception&) {
    // Listener closed (shutdown) or socket exhaustion; either way we stop
    // accepting.
  }
}

net::awaitable<void> HttpServer::Session(beast::tcp_stream stream) {
  // Buffers and requests are per-connection and reused across keep-alive
  // requests; only the body string is reallocated per request.
  beast::flat_buffer buffer;
  bool close = false;
  while (!close) {
    try {
      http::request<http::string_body> req;
      stream.expires_after(kReadTimeout);
      co_await http::async_read(stream, buffer, req, net::use_awaitable);

      const std::string target(req.target());
      close = !req.keep_alive();

      if (req.method() == http::verb::post && target == "/v1/completions") {
        if (req.body().size() > kMaxBodyBytes) {
          http::response<http::string_body> res = MakeResponse(
              req, http::status::payload_too_large, "application/json",
              MakeError(413, "request body too large").dump());
          co_await http::async_write(stream, res, net::use_awaitable);
          continue;
        }
        // Generation has no wall-clock bound; heartbeat detection is via
        // disconnects.
        stream.expires_never();
        co_await dispatcher_.HandleCompletions(stream, req);
      } else {
        int status = 0;
        std::string content_type = "application/json";
        std::string body;
        http::status code = http::status::not_found;
        if (dispatcher_.HandleAdmin(std::string(req.method_string()), target, status,
                                    content_type, body)) {
          code = static_cast<http::status>(status);
        } else {
          body = MakeError(404, "unknown endpoint: " + target).dump();
        }
        co_await http::async_write(
            stream, MakeResponse(req, code, std::move(content_type), std::move(body)),
            net::use_awaitable);
      }
    } catch (const std::exception& e) {
      // Read/write failure: the peer is gone or sent garbage. Ending the
      // session releases the connection and, via the completions handler's
      // cancel guard, the engine request. Log the reason: silent EOFs hide
      // real server bugs.
      std::fprintf(stderr, "inferx serve: session ended: %s\n", e.what());
      break;
    }
    // Pipelined leftovers, if any, belong to the next request's parse.
    if (buffer.size() > 0) buffer.clear();
  }
  boost::system::error_code ignored;
  stream.socket().shutdown(net::ip::tcp::socket::shutdown_both, ignored);
}

}  // namespace inferx::server
