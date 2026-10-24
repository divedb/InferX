#ifndef INFERX_SERVER_HTTP_SERVER_H_
#define INFERX_SERVER_HTTP_SERVER_H_

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

#include <boost/asio/any_io_executor.hpp>
#include <boost/asio/awaitable.hpp>

#include "inferx/server/connection.h"
#include "inferx/server/engine_client.h"
#include "inferx/server/middleware.h"
#include "inferx/server/router.h"
#include "inferx/server/server_config.h"
#include "inferx/server/services.h"
#include "inferx/server/status.h"

namespace inferx::server {

struct BoundEndpoint {
  std::string host;
  std::uint16_t port = 0;
  bool tls = false;
};

enum class DrainPolicy : std::uint8_t {
  kFinishActive,
  kCancelImmediately,
};

/// \brief Server lifecycle: validate, bind, accept, drain, stop.
///
/// `Start` must be called once, from a coroutine on the server executor
/// (the io_context the listener and sessions share). `BeginDrain` is
/// thread-safe and idempotent.
class HttpServer {
 public:
  HttpServer(ServerConfig config, std::shared_ptr<IEngineClient> engine,
             std::shared_ptr<IModelCatalog> models,
             std::shared_ptr<HealthService> health,
             std::shared_ptr<MetricsService> metrics,
             std::shared_ptr<AdmissionController> admission,
             std::shared_ptr<Router> router,
             std::shared_ptr<MiddlewareChain> middleware);
  ~HttpServer();

  HttpServer(const HttpServer&) = delete;
  HttpServer& operator=(const HttpServer&) = delete;
  HttpServer(HttpServer&&) = delete;
  HttpServer& operator=(HttpServer&&) = delete;

  /// Validates config, binds, starts accepting, and returns the bound
  /// address. Marks the server ready when no engine probe is required.
  [[nodiscard]] boost::asio::awaitable<Result<BoundEndpoint>> Start();

  /// Waits until a drain completes and all sessions are gone.
  [[nodiscard]] boost::asio::awaitable<void> Run();

  /// Thread-safe; idempotent. Stops accepting, marks readiness false, and
  /// applies the drain policy to active requests.
  void BeginDrain(DrainPolicy policy) noexcept;

  /// Idempotent. Call after Run() or BeginDrain().
  [[nodiscard]] boost::asio::awaitable<void> Stop();

  [[nodiscard]] BoundEndpoint local_endpoint() const;
  [[nodiscard]] bool serving() const noexcept;
  [[nodiscard]] std::size_t active_connections() const noexcept;
  [[nodiscard]] std::size_t active_requests() const noexcept;

 private:
  struct Impl;
  // Shared: the drain sequence spawned by BeginDrain keeps it alive even if
  // the HttpServer object is destroyed first.
  std::shared_ptr<Impl> impl_;
};

}  // namespace inferx::server

#endif  // INFERX_SERVER_HTTP_SERVER_H_
