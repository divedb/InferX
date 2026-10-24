#include "inferx/server/services.h"

#include <atomic>
#include <memory>
#include <mutex>
#include <optional>
#include <system_error>
#include <utility>

#include <boost/asio/co_spawn.hpp>
#include <boost/asio/redirect_error.hpp>
#include <boost/asio/steady_timer.hpp>
#include <boost/asio/this_coro.hpp>
#include <boost/asio/use_awaitable.hpp>

#include "inferx/server/api_error_mapper.h"
#include "inferx/server/engine_client.h"

namespace inferx::server {

// ---- StaticModelCatalog ----

std::vector<ModelInfo> StaticModelCatalog::FromConfig(
    const std::string& model_id, const std::vector<std::string>& served_names,
    std::optional<std::uint64_t> max_model_len) {
  ModelInfo primary;
  if (!served_names.empty()) {
    primary.id = served_names.front();
    for (std::size_t i = 1; i < served_names.size(); ++i) {
      primary.aliases.push_back(served_names[i]);
    }
  } else {
    primary.id = model_id;
  }
  if (primary.id.empty()) primary.id = model_id;
  if (primary.id != model_id && !model_id.empty()) {
    primary.aliases.push_back(model_id);
  }
  primary.owned_by = "inferx";
  primary.tasks = {ModelTask::kGenerate};
  primary.max_model_len = max_model_len;
  return {std::move(primary)};
}

StaticModelCatalog::StaticModelCatalog(std::vector<ModelInfo> models)
    : models_(std::move(models)) {}

boost::asio::awaitable<Result<ModelInfo>> StaticModelCatalog::Resolve(
    std::string_view requested_model, const TenantId& tenant) const {
  (void)tenant;
  for (const auto& model : models_) {
    if (model.id == requested_model) co_return model;
    for (const auto& alias : model.aliases) {
      if (alias == requested_model) co_return model;
    }
  }
  co_return ApiError{404,
                     ApiErrorKind::kNotFound,
                     "invalid_request_error",
                     "model_not_found",
                     "The model '" + std::string(requested_model) +
                         "' does not exist",
                     "model",
                     std::nullopt,
                     false,
                     {}};
}

boost::asio::awaitable<Result<std::vector<ModelInfo>>>
StaticModelCatalog::List(const TenantId& tenant) const {
  (void)tenant;
  co_return models_;
}

// ---- HealthService ----

HealthService::HealthService(std::shared_ptr<IEngineClient> engine)
    : engine_(std::move(engine)) {}

void HealthService::SetState(HealthState state, std::string detail) noexcept {
  state_.store(state, std::memory_order_release);
  {
    std::lock_guard<std::mutex> lock(detail_mutex_);
    detail_ = std::move(detail);
  }
}

HealthSnapshot HealthService::Snapshot() const noexcept {
  HealthSnapshot snapshot;
  snapshot.state = state_.load(std::memory_order_acquire);
  {
    std::lock_guard<std::mutex> lock(detail_mutex_);
    snapshot.detail = detail_;
  }
  // Without an engine wired, readiness follows the server lifecycle; with
  // one, /health must reflect the engine, not just this process.
  snapshot.engine_running = engine_ != nullptr;
  snapshot.engine_ready =
      engine_ == nullptr ? snapshot.state == HealthState::kReady : false;
  return snapshot;
}

boost::asio::awaitable<HealthSnapshot> HealthService::Probe(
    std::chrono::milliseconds timeout) {
  if (!engine_) {
    co_return Snapshot();
  }
  // Bounded probe: the engine call races a timer. Shared state keeps the
  // probe's completion handler safe even when the timer wins and this
  // coroutine returns first.
  struct ProbeState {
    std::optional<EngineHealth> health;
    std::exception_ptr error;
    bool done = false;
  };
  auto probe_state = std::make_shared<ProbeState>();
  auto engine = engine_;
  auto timer = std::make_shared<boost::asio::steady_timer>(
      co_await boost::asio::this_coro::executor);
  timer->expires_after(timeout);
  boost::asio::co_spawn(
      timer->get_executor(), [engine]() { return engine->CheckHealth(); },
      [probe_state, timer](std::exception_ptr error, EngineHealth health) {
        probe_state->error = std::move(error);
        probe_state->health = std::move(health);
        probe_state->done = true;
        timer->cancel();
      });
  boost::system::error_code wait_ec;
  co_await timer->async_wait(
      boost::asio::redirect_error(boost::asio::use_awaitable, wait_ec));

  HealthSnapshot snapshot;
  snapshot.state = state_.load(std::memory_order_acquire);
  if (probe_state->done && !probe_state->error) {
    snapshot.engine_running = probe_state->health->running;
    snapshot.engine_ready = probe_state->health->ready;
    snapshot.detail = std::move(probe_state->health->detail);
  } else {
    snapshot.engine_running = false;
    snapshot.engine_ready = false;
    snapshot.detail = "engine probe timed out or failed";
  }
  co_return snapshot;
}

// ---- AdmissionLease ----

struct AdmissionLease::State {
  std::atomic<std::size_t>* counter = nullptr;

  void Release() noexcept {
    if (counter == nullptr) return;
    const std::size_t previous =
        counter->fetch_sub(1, std::memory_order_acq_rel);
    (void)previous;
    counter = nullptr;
  }
};

AdmissionLease::~AdmissionLease() { Release(); }

AdmissionLease::AdmissionLease(AdmissionLease&& other) noexcept
    : state_(std::move(other.state_)) {}

AdmissionLease& AdmissionLease::operator=(AdmissionLease&& other) noexcept {
  if (this != &other) {
    Release();
    state_ = std::move(other.state_);
  }
  return *this;
}

void AdmissionLease::Release() noexcept {
  if (state_) state_->Release();
}

// ---- LeaseAdmissionController ----

LeaseAdmissionController::LeaseAdmissionController(
    AdmissionLimits limits, std::shared_ptr<HealthService> health)
    : limits_(limits), health_(std::move(health)) {}

boost::asio::awaitable<Result<AdmissionLease>> LeaseAdmissionController::Acquire(
    const AdmissionRequest& request, RequestContext& ctx) {
  (void)ctx;
  const HealthState state = health_->Snapshot().state;
  if (state == HealthState::kDraining || state == HealthState::kStopped) {
    co_return ApiErrorMapper::ShuttingDown();
  }
  if (request.body_bytes > limits_.max_body_bytes) {
    co_return ApiErrorMapper::PayloadTooLarge();
  }
  const std::size_t current = active_.fetch_add(1, std::memory_order_acq_rel);
  if (current >= limits_.max_active_requests) {
    active_.fetch_sub(1, std::memory_order_acq_rel);
    co_return ApiError{429,
                       ApiErrorKind::kRateLimit,
                       "rate_limit_error",
                       "rate_limit_exceeded",
                       "too many concurrent requests",
                       std::nullopt,
                       std::chrono::seconds(1),
                       false,
                       {}};
  }
  AdmissionLease lease;
  lease.state_ = std::make_shared<AdmissionLease::State>();
  lease.state_->counter = &active_;
  co_return lease;
}

std::size_t LeaseAdmissionController::active() const noexcept {
  return active_.load(std::memory_order_acquire);
}

// ---- CountingMetricsService ----

void CountingMetricsService::ObserveHttpRequest(const HttpRequestMetrics&) {
  http_requests_total_.fetch_add(1, std::memory_order_relaxed);
}
void CountingMetricsService::ObserveAdmissionRejected(std::string_view) {
  admission_rejections_total_.fetch_add(1, std::memory_order_relaxed);
}
void CountingMetricsService::ObserveGenerationStarted(std::string_view) {}
void CountingMetricsService::ObserveGenerationFinished(std::string_view,
                                                       FinishReason) {}
void CountingMetricsService::ObserveCancelled(std::string_view) {}
void CountingMetricsService::ObserveTtft(std::string_view,
                                         std::chrono::microseconds) {}
void CountingMetricsService::ObserveStreamBufferedBytes(std::size_t) {}
void CountingMetricsService::ObserveStreamWriteStall() {}
void CountingMetricsService::ObserveClientDisconnect() {
  client_disconnects_total_.fetch_add(1, std::memory_order_relaxed);
}
void CountingMetricsService::SetQueueDepth(std::size_t) {}
void CountingMetricsService::SetActiveRequests(std::size_t value) {
  active_requests_.store(value, std::memory_order_relaxed);
}

}  // namespace inferx::server
