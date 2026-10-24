#include "inferx/server/http_server.h"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <system_error>
#include <unordered_map>
#include <utility>
#include <vector>

#include <boost/asio/any_io_executor.hpp>
#include <boost/asio/awaitable.hpp>
#include <boost/asio/bind_executor.hpp>
#include <boost/asio/co_spawn.hpp>
#include <boost/asio/dispatch.hpp>
#include <boost/asio/ip/tcp.hpp>
#include <boost/asio/post.hpp>
#include <boost/asio/redirect_error.hpp>
#include <boost/asio/steady_timer.hpp>
#include <boost/asio/strand.hpp>
#include <boost/asio/use_awaitable.hpp>

#include "inferx/server/api_error_mapper.h"
#include "inferx/server/connection.h"

namespace inferx::server {

namespace net = boost::asio;
namespace beast = boost::beast;
using net::ip::tcp;

namespace {

std::atomic<std::uint64_t> g_next_connection_seq{1};

ConnectionId NextConnectionId() {
  return ConnectionId("conn-" + std::to_string(
      g_next_connection_seq.fetch_add(1, std::memory_order_relaxed)));
}

/// \brief Mutex-guarded registry: sessions live on their strands; Add/Remove
///        are called from arbitrary strands, queries from the server side.
class RegistryImpl final : public ConnectionRegistry {
 public:
  void Add(std::shared_ptr<HttpSession> session) override {
    const std::lock_guard<std::mutex> lock(mutex_);
    sessions_[session->id()] = std::move(session);
  }
  void Remove(ConnectionId id) noexcept override {
    const std::lock_guard<std::mutex> lock(mutex_);
    sessions_.erase(id);
  }
  std::size_t size() const noexcept override {
    const std::lock_guard<std::mutex> lock(mutex_);
    return sessions_.size();
  }
  std::size_t active_requests() const noexcept override { return 0; }
  void BeginDrainAll() noexcept override {
    for (auto& session : Snapshot()) session->BeginDrain();
  }
  net::awaitable<void> WaitEmpty(std::chrono::milliseconds timeout) override {
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    for (;;) {
      if (size() == 0) co_return;
      const auto now = std::chrono::steady_clock::now();
      if (now >= deadline) co_return;
      net::steady_timer timer(co_await net::this_coro::executor);
      timer.expires_at(std::min(now + std::chrono::milliseconds(10), deadline));
      boost::system::error_code ec;
      co_await timer.async_wait(net::redirect_error(net::use_awaitable, ec));
    }
  }
  /// Force-closes every remaining session (after the drain grace expired).
  void CloseAll() noexcept {
    for (auto& session : Snapshot()) session->Close();
  }

 private:
  std::vector<std::shared_ptr<HttpSession>> Snapshot() const {
    const std::lock_guard<std::mutex> lock(mutex_);
    std::vector<std::shared_ptr<HttpSession>> out;
    out.reserve(sessions_.size());
    for (const auto& [id, session] : sessions_) out.push_back(session);
    return out;
  }
  mutable std::mutex mutex_;
  std::unordered_map<ConnectionId, std::shared_ptr<HttpSession>,
                     StrongIdHash<ConnectionTag>>
      sessions_;
};

/// \brief Everything a session needs; shared so sessions provably outlive
///        the services they reference even if the HttpServer is destroyed.
struct SessionEnv {
  ServerConfig config;
  SessionServices services;
  std::shared_ptr<ShutdownState> shutdown;
  std::shared_ptr<RegistryImpl> registry;
};

}  // namespace

// ---- Listener ----

struct Listener::Impl {
  Impl(net::any_io_executor executor_in, const ServerConfig& config_in,
       ConnectionRegistry& registry_in, SessionFactory factory_in)
      : executor(std::move(executor_in)),
        config(config_in),
        registry(registry_in),
        factory(std::move(factory_in)),
        acceptor(executor) {}

  net::any_io_executor executor;
  ServerConfig config;
  ConnectionRegistry& registry;
  SessionFactory factory;
  tcp::acceptor acceptor;
  std::atomic<bool> listening{false};
  std::atomic<bool> closed{false};
};

Listener::Listener(net::any_io_executor executor, const ServerConfig& config,
                   ConnectionRegistry& registry, SessionFactory factory)
    : impl_(std::make_shared<Impl>(std::move(executor), config, registry,
                                   std::move(factory))) {}

Listener::~Listener() { Close(); }

net::awaitable<Result<tcp::endpoint>> Listener::Start() {
  boost::system::error_code ec;
  tcp::endpoint endpoint;
  if (impl_->config.bind_host.empty() || impl_->config.bind_host == "0.0.0.0") {
    endpoint = tcp::endpoint(tcp::v4(), impl_->config.bind_port);
  } else if (impl_->config.bind_host == "::") {
    endpoint = tcp::endpoint(tcp::v6(), impl_->config.bind_port);
  } else {
    boost::system::error_code resolve_ec;
    const auto results =
        tcp::resolver(impl_->executor)
            .resolve(impl_->config.bind_host,
                     std::to_string(impl_->config.bind_port), resolve_ec);
    if (resolve_ec || results.empty()) {
      co_return ApiErrorMapper::Internal(
          "bind_failed", "cannot resolve bind host: " + impl_->config.bind_host);
    }
    endpoint = results.begin()->endpoint();
  }

  auto& acceptor = impl_->acceptor;
  acceptor.open(endpoint.protocol(), ec);
  if (ec) co_return ApiErrorMapper::Internal("bind_failed", ec.message());
  acceptor.set_option(tcp::acceptor::reuse_address(true), ec);
  acceptor.bind(endpoint, ec);
  if (ec == net::error::address_in_use) {
    co_return ApiErrorMapper::Internal("address_in_use", ec.message());
  }
  if (ec) co_return ApiErrorMapper::Internal("bind_failed", ec.message());
  acceptor.listen(tcp::socket::max_listen_connections, ec);
  if (ec) co_return ApiErrorMapper::Internal("listen_failed", ec.message());
  impl_->listening.store(true, std::memory_order_release);

  const tcp::endpoint bound = acceptor.local_endpoint(ec);
  if (ec) co_return ApiErrorMapper::Internal("bind_failed", ec.message());

  // The accept loop keeps Impl alive through its own shared_ptr copy.
  net::co_spawn(
      impl_->executor,
      [impl = impl_]() -> net::awaitable<void> {
        co_await Listener::AcceptLoopOn(std::move(impl));
      },
      [](std::exception_ptr error) {
        if (error) {
          (void)ApiErrorMapper::FromException(error, ErrorOrigin::kInternal);
        }
      });
  co_return bound;
}

net::awaitable<void> Listener::AcceptLoopOn(std::shared_ptr<Impl> impl) {
  auto& acceptor = impl->acceptor;
  std::chrono::milliseconds backoff(10);
  for (;;) {
    if (impl->closed.load(std::memory_order_acquire)) co_return;
    net::strand<net::any_io_executor> strand(impl->executor);
    boost::system::error_code ec;
    tcp::socket socket =
        co_await acceptor.async_accept(strand, net::redirect_error(
            net::use_awaitable, ec));
    if (impl->closed.load(std::memory_order_acquire)) co_return;
    if (ec) {
      const bool transient =
          ec == net::error::connection_aborted || ec == net::error::interrupted ||
          ec == std::errc::too_many_files_open ||
          ec == std::errc::too_many_files_open_in_system;
      if (!transient) co_return;  // acceptor closed or fatal: stop quietly
      // Overload pause: never spin; exponential backoff capped at 1s.
      net::steady_timer timer(co_await net::this_coro::executor);
      timer.expires_after(backoff);
      boost::system::error_code wait_ec;
      co_await timer.async_wait(
          net::redirect_error(net::use_awaitable, wait_ec));
      backoff = std::min(backoff * 2, std::chrono::milliseconds(1000));
      continue;
    }
    backoff = std::chrono::milliseconds(10);

    if (impl->registry.size() >= impl->config.http.max_connections) {
      // At the ceiling: accept then close immediately so the client sees a
      // reset rather than a listen-queue stall.
      boost::system::error_code close_ec;
      socket.close(close_ec);
      continue;
    }
    boost::system::error_code option_ec;
    socket.set_option(tcp::no_delay(true), option_ec);

    ConnectionContext context;
    context.id = NextConnectionId();
    context.tls = false;
    const auto remote = socket.remote_endpoint(option_ec);
    if (!option_ec) {
      context.peer_ip = remote.address().to_string();
      context.peer_port = remote.port();
    }
    beast::tcp_stream stream(std::move(socket));
    std::shared_ptr<HttpSession> session =
        impl->factory(std::move(stream), std::move(context));
    impl->registry.Add(session);
    session->Start();
  }
}

void Listener::Close() noexcept {
  if (impl_->closed.exchange(true, std::memory_order_acq_rel)) return;
  impl_->listening.store(false, std::memory_order_release);
  auto acceptor = &impl_->acceptor;
  net::post(impl_->executor, [acceptor]() {
    boost::system::error_code ec;
    if (acceptor->is_open()) acceptor->close(ec);
  });
}

bool Listener::listening() const noexcept {
  return impl_->listening.load(std::memory_order_acquire);
}

// ---- HttpServer ----

struct HttpServer::Impl {
  enum class State { kCreated, kServing, kDraining, kStopped };

  Impl(ServerConfig config_in, std::shared_ptr<IEngineClient> engine_in,
       std::shared_ptr<IModelCatalog> models_in,
       std::shared_ptr<HealthService> health_in,
       std::shared_ptr<MetricsService> metrics_in,
       std::shared_ptr<AdmissionController> admission_in,
       std::shared_ptr<Router> router_in,
       std::shared_ptr<MiddlewareChain> middleware_in)
      : config(std::move(config_in)),
        env(std::make_shared<SessionEnv>(SessionEnv{
              config,
              SessionServices{std::move(engine_in), std::move(models_in),
                              std::move(health_in), std::move(metrics_in),
                              std::move(admission_in), std::move(router_in),
                              std::move(middleware_in)},
              std::make_shared<ShutdownState>(),
              std::make_shared<RegistryImpl>()})) {}

  ServerConfig config;
  std::shared_ptr<SessionEnv> env;
  net::any_io_executor executor;
  std::unique_ptr<net::steady_timer> stop_signal;
  std::unique_ptr<Listener> listener_owner;
  std::atomic<State> state{State::kCreated};
  BoundEndpoint bound;
  std::atomic<bool> drain_started{false};
};

HttpServer::HttpServer(ServerConfig config,
                       std::shared_ptr<IEngineClient> engine,
                       std::shared_ptr<IModelCatalog> models,
                       std::shared_ptr<HealthService> health,
                       std::shared_ptr<MetricsService> metrics,
                       std::shared_ptr<AdmissionController> admission,
                       std::shared_ptr<Router> router,
                       std::shared_ptr<MiddlewareChain> middleware)
    : impl_(std::make_shared<Impl>(std::move(config), std::move(engine),
                                   std::move(models), std::move(health),
                                   std::move(metrics), std::move(admission),
                                   std::move(router), std::move(middleware))) {}

HttpServer::~HttpServer() = default;

net::awaitable<Result<BoundEndpoint>> HttpServer::Start() {
  if (auto valid = ValidateConfig(impl_->config); !valid) {
    co_return valid.error();
  }
  impl_->executor = co_await net::this_coro::executor;

  auto env = impl_->env;
  auto listener = std::make_unique<Listener>(
      impl_->executor, impl_->config, *env->registry,
      [env](beast::tcp_stream&& stream, ConnectionContext ctx) {
        ctx.shutdown = env->shutdown;
        return std::make_shared<HttpSession>(std::move(stream), std::move(ctx),
                                             env->config, env->services,
                                             *env->registry);
      });
  auto started = co_await listener->Start();
  if (!started) co_return started.error();
  impl_->listener_owner = std::move(listener);
  const tcp::endpoint& endpoint = started.value();
  impl_->bound =
      BoundEndpoint{endpoint.address().to_string(), endpoint.port(),
                    impl_->config.tls.enabled};
  impl_->stop_signal = std::make_unique<net::steady_timer>(impl_->executor);
  impl_->stop_signal->expires_at(
      (std::chrono::steady_clock::time_point::max)());
  impl_->state.store(Impl::State::kServing, std::memory_order_release);
  impl_->env->services.health->SetState(HealthState::kReady);
  co_return impl_->bound;
}

net::awaitable<void> HttpServer::Run() {
  // Parks until the drain sequence cancels the timer.
  if (impl_->stop_signal) {
    boost::system::error_code ec;
    co_await impl_->stop_signal->async_wait(
        net::redirect_error(net::use_awaitable, ec));
  }
}

void HttpServer::BeginDrain(DrainPolicy policy) noexcept {
  (void)policy;  // kCancelImmediately tightens the grace in later phases
  if (impl_->drain_started.exchange(true, std::memory_order_acq_rel)) return;
  auto impl = impl_;
  net::dispatch(impl->executor, [impl]() {
    net::co_spawn(
        impl->executor,
        [impl]() -> net::awaitable<void> {
          const auto current = impl->state.load(std::memory_order_acquire);
          if (current == Impl::State::kStopped ||
              current == Impl::State::kDraining) {
            co_return;
          }
          impl->state.store(Impl::State::kDraining, std::memory_order_release);
          impl->env->shutdown->draining.store(true, std::memory_order_release);
          impl->env->shutdown->stop_accepting.store(true,
                                                    std::memory_order_release);
          impl->env->services.health->SetState(HealthState::kDraining,
                                               "draining");
          if (impl->listener_owner) impl->listener_owner->Close();
          impl->env->registry->BeginDrainAll();
          co_await impl->env->registry->WaitEmpty(
              impl->config.timeouts.drain_grace);
          // Grace expired: force-close whatever remains, then wait briefly
          // for sessions to deregister.
          impl->env->registry->CloseAll();
          co_await impl->env->registry->WaitEmpty(
              impl->config.timeouts.cancel_grace);
          impl->state.store(Impl::State::kStopped, std::memory_order_release);
          impl->env->services.health->SetState(HealthState::kStopped,
                                               "stopped");
          if (impl->stop_signal) impl->stop_signal->cancel();
        },
        [](std::exception_ptr error) {
          if (error) {
            (void)ApiErrorMapper::FromException(error, ErrorOrigin::kInternal);
          }
        });
  });
}

net::awaitable<void> HttpServer::Stop() {
  if (impl_->state.load(std::memory_order_acquire) == Impl::State::kCreated) {
    co_return;  // never started: nothing to drain
  }
  BeginDrain(DrainPolicy::kFinishActive);
  while (impl_->state.load(std::memory_order_acquire) !=
         Impl::State::kStopped) {
    net::steady_timer timer(co_await net::this_coro::executor);
    timer.expires_after(std::chrono::milliseconds(5));
    boost::system::error_code ec;
    co_await timer.async_wait(net::redirect_error(net::use_awaitable, ec));
  }
  // Wake any Run() awaiter parked on the stop signal. Never await it here:
  // the drain sequence already cancelled it, and a wait registered after
  // that cancel would block until the max expiry.
  if (impl_->stop_signal) impl_->stop_signal->cancel();
}

BoundEndpoint HttpServer::local_endpoint() const { return impl_->bound; }

bool HttpServer::serving() const noexcept {
  return impl_->state.load(std::memory_order_acquire) ==
         Impl::State::kServing;
}

std::size_t HttpServer::active_connections() const noexcept {
  return impl_->env->registry->size();
}

std::size_t HttpServer::active_requests() const noexcept {
  return impl_->env->registry->active_requests();
}

}  // namespace inferx::server
