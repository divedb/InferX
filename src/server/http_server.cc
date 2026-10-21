// HttpServer implementation: listener + per-connection session coroutines
// (see http_server.h). Conventions: every response declares keep-alive per
// the request; socket errors end the session quietly (disconnects are
// routine, not exceptional); detached coroutines never let exceptions
// escape (that would terminate the process).
#include "inferx/server/http_server.h"

#include <cstdio>
#include <memory>
#include <utility>

#include "inferx/core/logging.h"
#include "inferx/server/api.h"

namespace inferx {
namespace {

// A request head must arrive within this window; generation itself is not
// bounded (expires_never while a handler owns the stream).
constexpr auto kReadTimeout = std::chrono::seconds(30);
// Requests larger than this are rejected before parsing.
constexpr std::size_t kMaxBodyBytes = 8u << 20;

bool IsHealthProbe(const std::string& path) {
  return path == "/health" || path == "/livez" || path == "/readyz";
}

}  // namespace

HttpServer::HttpServer(net::io_context& io, std::string host, std::uint16_t port,
                       RequestHandler& handler)
    : io_(io), host_(std::move(host)), port_(port), handler_(handler) {}

std::uint16_t HttpServer::Listen() {
  auto acceptor = std::make_shared<net::ip::tcp::acceptor>(io_.get_executor());
  net::ip::tcp::endpoint endpoint(net::ip::make_address(host_), port_);
  acceptor->open(endpoint.protocol());
  acceptor->set_option(net::socket_base::reuse_address(true));
  acceptor->bind(endpoint);
  acceptor->listen(net::socket_base::max_listen_connections);
  acceptor_ = acceptor;
  const std::uint16_t bound =
      acceptor->local_endpoint().port();
  net::co_spawn(io_, ListenLoop(), net::detached);
  return bound;
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

void HttpServer::SetDraining(bool draining) { draining_->store(draining); }

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

      close = !req.keep_alive();
      const std::string target(req.target());
      const std::size_t query = target.find('?');
      const std::string path = query == std::string::npos ? target : target.substr(0, query);

      // Drain mode: health probes keep answering (with their real state);
      // everything else is a 503 with Connection: close.
      if (draining_->load() && !IsHealthProbe(path)) {
        http::response<http::string_body> res{
            http::status::service_unavailable, req.version()};
        res.set(http::field::server, "inferx");
        res.set(http::field::content_type, "application/json");
        res.set("Connection", "close");
        res.keep_alive(false);
        res.body() = MakeError(503, "the server is draining and not accepting new requests",
                               "", "shutting_down")
                         .dump();
        res.prepare_payload();
        co_await http::async_write(stream, res, net::use_awaitable);
        break;
      }

      if (req.method() == http::verb::post && req.body().size() > kMaxBodyBytes) {
        http::response<http::string_body> res{
            http::status::payload_too_large, req.version()};
        res.set(http::field::server, "inferx");
        res.set(http::field::content_type, "application/json");
        res.keep_alive(req.keep_alive());
        res.body() = MakeError(413, "request body exceeds the configured limit", "",
                               "payload_too_large")
                         .dump();
        res.prepare_payload();
        co_await http::async_write(stream, res, net::use_awaitable);
        continue;
      }

      // Handlers own the stream while responding (SSE has no wall-clock
      // bound; disconnects surface as write errors).
      const std::string method(req.method_string());
      const bool exchange_keep_alive = req.keep_alive();
      stream.expires_never();
      HttpExchange exchange{std::move(req), method, path, target,
                            ResponseWriter(stream, exchange_keep_alive)};

      const bool handled = co_await handler_.Handle(exchange);
      if (!handled && !exchange.writer.committed()) {
        http::response<http::string_body> res{
            http::status::not_found, exchange.request.version()};
        res.set(http::field::server, "inferx");
        res.set(http::field::content_type, "application/json");
        res.keep_alive(exchange.request.keep_alive());
        res.body() = MakeError(404, "unknown endpoint: " + exchange.target).dump();
        res.prepare_payload();
        co_await http::async_write(stream, res, net::use_awaitable);
      }
      if (exchange.writer.failed()) break;
    } catch (const std::exception& e) {
      // Read/write failure: the peer is gone or sent garbage. Ending the
      // session releases the connection and, via the handler's cancel
      // guard, the engine request. Log the reason: silent EOFs hide real
      // server bugs.
      INFERX_LOG(WARNING) << "inferx serve: session ended: " << e.what();
      break;
    }
    // Pipelined leftovers, if any, belong to the next request's parse.
    if (buffer.size() > 0) buffer.clear();
  }
  boost::system::error_code ignored;
  stream.socket().shutdown(net::ip::tcp::socket::shutdown_both, ignored);
}

}  // namespace inferx
