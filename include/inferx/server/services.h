#ifndef INFERX_SERVER_SERVICES_H_
#define INFERX_SERVER_SERVICES_H_

#include <atomic>
#include <chrono>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include <boost/asio/awaitable.hpp>

#include "inferx/server/engine_types.h"
#include "inferx/server/request_context.h"
#include "inferx/server/status.h"

namespace inferx::server {

class IEngineClient;

/// \brief Catalog entry describing one served model and its capabilities.
struct ModelInfo {
  std::string id;
  std::string owned_by;
  std::vector<ModelTask> tasks;
  std::optional<std::uint64_t> max_model_len;
  std::vector<std::string> aliases;
  bool supports_streaming = true;
  bool supports_tools = false;
  bool supports_vision = false;
  bool supports_audio = false;
  bool supports_video = false;
};

/// \brief Alias resolution and listing. Queryable at startup; stable for the
///        process lifetime in the MVP (no hot model swap).
class IModelCatalog {
 public:
  virtual ~IModelCatalog() = default;
  virtual boost::asio::awaitable<Result<ModelInfo>> Resolve(
      std::string_view requested_model, const TenantId& tenant) const = 0;
  virtual boost::asio::awaitable<Result<std::vector<ModelInfo>>> List(
      const TenantId& tenant) const = 0;
};

/// \brief Static catalog from ServerConfig: one served model plus aliases.
class StaticModelCatalog final : public IModelCatalog {
 public:
  /// `names[0]` (or `model_id` when empty) is the primary ID; the rest are
  /// aliases.
  static std::vector<ModelInfo> FromConfig(
      const std::string& model_id, const std::vector<std::string>& served_names,
      std::optional<std::uint64_t> max_model_len);

  explicit StaticModelCatalog(std::vector<ModelInfo> models);

  boost::asio::awaitable<Result<ModelInfo>> Resolve(
      std::string_view requested_model, const TenantId& tenant) const override;
  boost::asio::awaitable<Result<std::vector<ModelInfo>>> List(
      const TenantId& tenant) const override;

 private:
  std::vector<ModelInfo> models_;
};

enum class HealthState : std::uint8_t {
  kStarting,
  kReady,
  kDraining,
  kStopped,
};

struct HealthSnapshot {
  HealthState state = HealthState::kStarting;
  bool engine_running = false;
  bool engine_ready = false;
  std::string detail;
};

/// \brief Liveness/readiness snapshots and engine probes. `/health` uses a
///        cached snapshot; explicit probes carry a hard timeout.
class HealthService {
 public:
  /// `engine` may be null (no engine wired yet): snapshots then reflect only
  /// the server lifecycle state and `engine_ready` follows `state == kReady`.
  explicit HealthService(std::shared_ptr<IEngineClient> engine);

  virtual ~HealthService() = default;

  virtual void SetState(HealthState state, std::string detail = {}) noexcept;
  [[nodiscard]] virtual HealthSnapshot Snapshot() const noexcept;

  /// Probes the engine with a timeout when one is wired; otherwise returns
  /// the cached snapshot. Never blocks beyond `timeout`.
  [[nodiscard]] virtual boost::asio::awaitable<HealthSnapshot> Probe(
      std::chrono::milliseconds timeout);

 private:
  std::shared_ptr<IEngineClient> engine_;
  std::atomic<HealthState> state_{HealthState::kStarting};
  mutable std::mutex detail_mutex_;
  std::string detail_;
};

struct AdmissionLimits {
  std::size_t max_connections = 4096;
  std::size_t max_active_requests = 1024;
  std::chrono::milliseconds max_queue_wait{5000};
  std::size_t max_body_bytes = 8u * 1024u * 1024u;
  std::size_t max_media_bytes = 16u * 1024u * 1024u;
  std::size_t max_media_items = 8;
};

struct AdmissionRequest {
  TenantId tenant;
  std::string api_key_id;
  std::size_t body_bytes = 0;
  std::size_t media_bytes = 0;
  std::size_t media_items = 0;
  std::uint32_t sequences = 1;
  std::optional<std::uint64_t> estimated_prompt_tokens;
  std::string route_operation_id;
};

/// \brief Move-only lease. Destruction releases the active slot exactly once.
class AdmissionLease {
 public:
  AdmissionLease() = default;
  ~AdmissionLease();
  AdmissionLease(const AdmissionLease&) = delete;
  AdmissionLease& operator=(const AdmissionLease&) = delete;
  AdmissionLease(AdmissionLease&&) noexcept;
  AdmissionLease& operator=(AdmissionLease&&) noexcept;
  void Release() noexcept;

 private:
  friend class LeaseAdmissionController;
  struct State;
  std::shared_ptr<State> state_;
};

/// \brief Local admission control: body limits, active-request ceiling, and
///        the draining gate. Phase 1 rejects immediately (429 full / 503
///        draining); bounded FIFO queueing arrives with engine-backed phases.
class AdmissionController {
 public:
  virtual ~AdmissionController() = default;

  /// May wait in a bounded FIFO for `max_queue_wait`, honoring request
  /// cancellation. Returns 429 when full/expired, 503 when draining.
  [[nodiscard]] virtual boost::asio::awaitable<Result<AdmissionLease>>
  Acquire(const AdmissionRequest& request, RequestContext& ctx) = 0;

  [[nodiscard]] virtual std::size_t queued() const noexcept = 0;
  [[nodiscard]] virtual std::size_t active() const noexcept = 0;
};

/// \brief Lease-based AdmissionController over HealthService drain state.
class LeaseAdmissionController final : public AdmissionController {
 public:
  LeaseAdmissionController(AdmissionLimits limits,
                           std::shared_ptr<HealthService> health);

  boost::asio::awaitable<Result<AdmissionLease>> Acquire(
      const AdmissionRequest& request, RequestContext& ctx) override;

  [[nodiscard]] std::size_t queued() const noexcept override { return 0; }
  [[nodiscard]] std::size_t active() const noexcept override;

 private:
  AdmissionLimits limits_;
  std::shared_ptr<HealthService> health_;
  std::atomic<std::size_t> active_{0};
};

struct HttpRequestMetrics {
  std::string route;
  std::string method;
  int status = 0;
  std::chrono::microseconds duration{0};
  std::size_t request_bytes = 0;
  std::size_t response_bytes = 0;
  bool streamed = false;
};

/// \brief Low-cardinality metric sink. Phase 1 keeps in-process counters; a
///        Prometheus text exporter is a later optional tier.
class MetricsService {
 public:
  virtual ~MetricsService() = default;
  virtual void ObserveHttpRequest(const HttpRequestMetrics& metrics) = 0;
  virtual void ObserveAdmissionRejected(std::string_view reason) = 0;
  virtual void ObserveGenerationStarted(std::string_view model) = 0;
  virtual void ObserveGenerationFinished(std::string_view model,
                                         FinishReason reason) = 0;
  virtual void ObserveCancelled(std::string_view reason) = 0;
  virtual void ObserveTtft(std::string_view source,
                           std::chrono::microseconds value) = 0;
  virtual void ObserveStreamBufferedBytes(std::size_t bytes) = 0;
  virtual void ObserveStreamWriteStall() = 0;
  virtual void ObserveClientDisconnect() = 0;
  virtual void SetQueueDepth(std::size_t value) = 0;
  virtual void SetActiveRequests(std::size_t value) = 0;
};

/// \brief Counting implementation; thread-safe.
class CountingMetricsService final : public MetricsService {
 public:
  void ObserveHttpRequest(const HttpRequestMetrics& metrics) override;
  void ObserveAdmissionRejected(std::string_view reason) override;
  void ObserveGenerationStarted(std::string_view model) override;
  void ObserveGenerationFinished(std::string_view model,
                                 FinishReason reason) override;
  void ObserveCancelled(std::string_view reason) override;
  void ObserveTtft(std::string_view source,
                   std::chrono::microseconds value) override;
  void ObserveStreamBufferedBytes(std::size_t bytes) override;
  void ObserveStreamWriteStall() override;
  void ObserveClientDisconnect() override;
  void SetQueueDepth(std::size_t value) override;
  void SetActiveRequests(std::size_t value) override;

  std::uint64_t http_requests_total() const noexcept {
    return http_requests_total_.load(std::memory_order_relaxed);
  }

 private:
  std::atomic<std::uint64_t> http_requests_total_{0};
  std::atomic<std::uint64_t> admission_rejections_total_{0};
  std::atomic<std::uint64_t> client_disconnects_total_{0};
  std::atomic<std::size_t> active_requests_{0};
  mutable std::mutex mutex_;
};

}  // namespace inferx::server

#endif  // INFERX_SERVER_SERVICES_H_
