#include "inferx/server/transport/beast_listener.h"

#include <boost/asio/dispatch.hpp>
#include <boost/asio/ip/address.hpp>
#include <boost/asio/ip/tcp.hpp>
#include <boost/asio/strand.hpp>

#include <algorithm>
#include <atomic>
#include <memory>
#include <mutex>
#include <utility>
#include <vector>

#include "inferx/server/transport/beast_session.h"

namespace inferx::server::transport {
namespace {

namespace asio = boost::asio;
using tcp = asio::ip::tcp;

}  // namespace

struct BeastListener::Impl : std::enable_shared_from_this<Impl> {
  Impl(IoRuntime* runtime_value, BeastListenerConfig config_value,
       std::shared_ptr<RequestHandler> handler_value)
      : runtime(runtime_value),
        config(std::move(config_value)),
        handler(std::move(handler_value)),
        acceptor(runtime->accept_context()) {}

  IoRuntime* runtime;
  BeastListenerConfig config;
  std::shared_ptr<RequestHandler> handler;
  tcp::acceptor acceptor;
  std::atomic<bool> started{false};
  std::atomic<bool> stopping{false};
  std::atomic<int> bound_port{0};
  std::mutex sessions_mutex;
  std::vector<std::weak_ptr<BeastSession>> sessions;

  void Accept() {
    if (stopping.load(std::memory_order_acquire)) return;
    auto self = shared_from_this();
    auto socket = std::make_shared<tcp::socket>(
        asio::make_strand(runtime->NextContext()));
    acceptor.async_accept(*socket, [self, socket](boost::system::error_code error) {
      if (!error && !self->stopping.load(std::memory_order_acquire)) {
        auto session = MakeBeastSession(
            std::move(*socket), self->config, self->handler,
            self->runtime->coroutine_executor());
        {
          std::lock_guard lock(self->sessions_mutex);
          self->sessions.erase(
              std::remove_if(self->sessions.begin(), self->sessions.end(),
                             [](const auto& weak) { return weak.expired(); }),
              self->sessions.end());
          self->sessions.emplace_back(session);
        }
        StartBeastSession(session);
      }
      if (!self->stopping.load(std::memory_order_acquire)) self->Accept();
    });
  }
};

BeastListener::BeastListener(std::shared_ptr<Impl> impl)
    : impl_(std::move(impl)) {}

BeastListener::~BeastListener() { Stop(); }

StatusOr<std::unique_ptr<BeastListener>> BeastListener::Create(
    IoRuntime* runtime, BeastListenerConfig config,
    std::shared_ptr<RequestHandler> handler) {
  if (runtime == nullptr) return InvalidArgumentError("I/O runtime is null");
  if (handler == nullptr) return InvalidArgumentError("request handler is null");
  if (config.port < 0 || config.port > 65535) {
    return InvalidArgumentError("port must be in [0, 65535]");
  }
  if (config.max_request_bytes == 0 ||
      config.header_read_timeout_seconds <= 0 ||
      config.body_read_timeout_seconds <= 0 ||
      config.write_timeout_seconds <= 0 ||
      config.keep_alive_timeout_seconds <= 0) {
    return InvalidArgumentError("transport limits and timeouts must be positive");
  }
  return std::unique_ptr<BeastListener>(new BeastListener(
      std::make_shared<Impl>(runtime, std::move(config), std::move(handler))));
}

Status BeastListener::Start() {
  if (impl_->started.exchange(true, std::memory_order_acq_rel)) {
    return FailedPreconditionError("listener already started");
  }
  boost::system::error_code error;
  const auto address = asio::ip::make_address(impl_->config.host, error);
  if (error) return InvalidArgumentError("invalid listen address: ",
                                         impl_->config.host);
  const tcp::endpoint endpoint(
      address, static_cast<unsigned short>(impl_->config.port));
  impl_->acceptor.open(endpoint.protocol(), error);
  if (error) return InternalError("cannot open listener: ", error.message());
  impl_->acceptor.set_option(asio::socket_base::reuse_address(true), error);
  if (error) return InternalError("cannot configure listener: ", error.message());
  impl_->acceptor.bind(endpoint, error);
  if (error) return InternalError("cannot bind listener: ", error.message());
  impl_->acceptor.listen(asio::socket_base::max_listen_connections, error);
  if (error) return InternalError("cannot listen: ", error.message());
  impl_->bound_port.store(impl_->acceptor.local_endpoint(error).port(),
                          std::memory_order_release);
  if (error) return InternalError("cannot inspect listener: ", error.message());
  impl_->Accept();
  return OkStatus();
}

void BeastListener::Stop() {
  if (impl_->stopping.exchange(true, std::memory_order_acq_rel)) return;
  std::vector<std::shared_ptr<BeastSession>> active;
  {
    std::lock_guard lock(impl_->sessions_mutex);
    for (const auto& weak : impl_->sessions) {
      if (auto session = weak.lock()) active.push_back(std::move(session));
    }
    impl_->sessions.clear();
  }
  asio::dispatch(impl_->acceptor.get_executor(), [impl = impl_] {
    boost::system::error_code ignored;
    impl->acceptor.cancel(ignored);
    impl->acceptor.close(ignored);
  });
  for (const auto& session : active) StopBeastSession(session);
}

int BeastListener::port() const {
  return impl_->bound_port.load(std::memory_order_acquire);
}

}  // namespace inferx::server::transport
