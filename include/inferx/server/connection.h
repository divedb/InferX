#ifndef INFERX_SERVER_CONNECTION_H_
#define INFERX_SERVER_CONNECTION_H_

#include <atomic>
#include <chrono>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>

#include <boost/asio/any_io_executor.hpp>
#include <boost/asio/awaitable.hpp>
#include <boost/asio/ip/tcp.hpp>
#include <boost/beast/core/tcp_stream.hpp>

#include "inferx/server/engine_client.h"
#include "inferx/server/http_types.h"
#include "inferx/server/middleware.h"
#include "inferx/server/router.h"
#include "inferx/server/server_config.h"
#include "inferx/server/services.h"
#include "inferx/server/status.h"

namespace inferx::server {

/// \brief Process-wide shutdown flags shared with every connection.
struct ShutdownState {
  std::atomic<bool> draining{false};
  std::atomic<bool> stop_accepting{false};
};

struct ConnectionContext {
  ConnectionId id;
  std::string peer_ip;
  std::uint16_t peer_port = 0;
  bool tls = false;
  std::shared_ptr<ShutdownState> shutdown;
};

/// \brief Services injected into every session; all must outlive the
///        sessions (HttpServer destroys sessions before releasing them).
struct SessionServices {
  std::shared_ptr<IEngineClient> engine;
  std::shared_ptr<IModelCatalog> models;
  std::shared_ptr<HealthService> health;
  std::shared_ptr<MetricsService> metrics;
  std::shared_ptr<AdmissionController> admission;
  std::shared_ptr<Router> router;
  std::shared_ptr<MiddlewareChain> middleware;
};

class HttpSession;

/// \brief Tracks live sessions and active request count for shutdown.
class ConnectionRegistry {
 public:
  virtual ~ConnectionRegistry() = default;
  virtual void Add(std::shared_ptr<HttpSession> session) = 0;
  virtual void Remove(ConnectionId id) noexcept = 0;
  [[nodiscard]] virtual std::size_t size() const noexcept = 0;
  [[nodiscard]] virtual std::size_t active_requests() const noexcept = 0;
  virtual void BeginDrainAll() noexcept = 0;
  virtual boost::asio::awaitable<void> WaitEmpty(
      std::chrono::milliseconds timeout) = 0;
};

/// \brief One HTTP/1.1 connection: parse, limits, dispatch, keep-alive.
///
/// One outstanding request per connection; responses are written
/// sequentially. All state transitions happen on the session strand.
class HttpSession : public std::enable_shared_from_this<HttpSession> {
 public:
  HttpSession(boost::beast::tcp_stream&& stream, ConnectionContext context,
              ServerConfig config, SessionServices services,
              ConnectionRegistry& registry);
  ~HttpSession();

  HttpSession(const HttpSession&) = delete;
  HttpSession& operator=(const HttpSession&) = delete;

  /// Starts the session coroutine. Non-blocking; called by the acceptor.
  void Start();

  /// Drain-safe: stops reading new requests and closes the socket after the
  /// in-flight response terminates. Idempotent; posts to the strand.
  void BeginDrain() noexcept;

  /// Closes the socket immediately. Idempotent; posts to the strand.
  void Close() noexcept;

  [[nodiscard]] ConnectionId id() const noexcept;

 private:
  void PostStrand(std::function<void()> action) noexcept;
  boost::asio::awaitable<void> Run();
  boost::asio::awaitable<void> WriteResponse(HttpResponse response,
                                             bool keep_alive, bool head_only);
  boost::asio::awaitable<void> WriteEarlyError(const ApiError& error);
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

/// \brief The acceptor always produces a TCP stream. When TLS is enabled
///        (later phase), the session wraps it and performs the handshake
///        before HTTP parsing; ConnectionContext::tls records the active
///        path.
using SessionFactory = std::function<std::shared_ptr<HttpSession>(
    boost::beast::tcp_stream&&, ConnectionContext)>;

/// \brief Bind/listen/accept loop with transient-error backoff and the
///        connection ceiling.
class Listener {
 public:
  Listener(boost::asio::any_io_executor executor, const ServerConfig& config,
           ConnectionRegistry& registry, SessionFactory factory);
  ~Listener();

  Listener(const Listener&) = delete;
  Listener& operator=(const Listener&) = delete;

  /// Binds, listens, and starts the accept loop. The returned endpoint is
  /// the actual bound address (useful when port 0 is requested).
  [[nodiscard]] boost::asio::awaitable<Result<boost::asio::ip::tcp::endpoint>>
  Start();

  void Close() noexcept;
  [[nodiscard]] bool listening() const noexcept;

 private:
  struct Impl;
  /// The accept loop holds a copy, so it outlives the Listener object while
  /// remaining reachable by Close().
  static boost::asio::awaitable<void> AcceptLoopOn(std::shared_ptr<Impl> impl);
  std::shared_ptr<Impl> impl_;
};

}  // namespace inferx::server

#endif  // INFERX_SERVER_CONNECTION_H_
