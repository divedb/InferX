#pragma once

#include <atomic>

#include "inferx/server/transport/response_writer.h"

namespace inferx::server::handlers {

/// Thread-safe process health inputs maintained by the composition root.
class HealthState {
 public:
  void SetEventLoopHealthy(bool value) { event_loop_healthy_.store(value); }
  void SetCriticalThreadsHealthy(bool value) {
    critical_threads_healthy_.store(value);
  }
  void SetConfigurationLoaded(bool value) {
    configuration_loaded_.store(value);
  }
  void SetDependenciesLoaded(bool value) { dependencies_loaded_.store(value); }
  void SetListenerAccepting(bool value) { listener_accepting_.store(value); }
  void SetSchedulerConnected(bool value) { scheduler_connected_.store(value); }
  void SetReadyModelAvailable(bool value) {
    ready_model_available_.store(value);
  }

  bool live() const;
  bool startup() const;
  bool ready() const;

 private:
  std::atomic<bool> event_loop_healthy_{true};
  std::atomic<bool> critical_threads_healthy_{true};
  std::atomic<bool> configuration_loaded_{false};
  std::atomic<bool> dependencies_loaded_{false};
  std::atomic<bool> listener_accepting_{false};
  std::atomic<bool> scheduler_connected_{false};
  std::atomic<bool> ready_model_available_{false};
};

enum class HealthProbe { kLive, kReady, kStartup };

class HealthHandler final : public transport::RequestHandler {
 public:
  HealthHandler(const HealthState* state, HealthProbe probe)
      : state_(state), probe_(probe) {}

  folly::coro::Task<void> Handle(
      transport::HttpRequest request, transport::ResponseWriter& response,
      folly::CancellationToken cancellation) override;

 private:
  const HealthState* state_;
  HealthProbe probe_;
};

}  // namespace inferx::server::handlers
