// MetricsService implementation (see metrics.h).
#include "inferx/server/metrics.h"

#include <algorithm>
#include <mutex>
#include <utility>

namespace inferx {

void MetricsService::Increment(std::string_view name, std::uint64_t by) {
  std::lock_guard<std::mutex> lock(mu_);
  counters_[std::string(name)] += by;
}

void MetricsService::SetGauge(std::string_view name, double value) {
  std::lock_guard<std::mutex> lock(mu_);
  gauges_[std::string(name)] = value;
}

void MetricsService::ObserveRequest(const std::string& route, const std::string& method,
                                    int status) {
  Increment("inferx_http_requests_total{route=\"" + route + "\",method=\"" + method +
            "\",status=\"" + std::to_string(status) + "\"}");
}

void MetricsService::ObservePromptRejected(std::string_view reason) {
  Increment("inferx_prompt_rejected_total{reason=\"" + std::string(reason) + "\"}");
}

void MetricsService::ObserveDisconnect() {
  Increment("inferx_client_disconnects_total");
}

void MetricsService::ObserveEngineAbort() {
  Increment("inferx_engine_aborts_total");
}

void MetricsService::ObservePoolHealth(const PoolHealth& health) {
  SetGauge("inferx_prompt_workers_ready", health.workers_ready);
  SetGauge("inferx_prompt_workers_busy", health.workers_busy);
  SetGauge("inferx_prompt_workers_down", health.workers_down);
  SetGauge("inferx_prompt_queued_requests",
           static_cast<double>(health.queued_requests));
  SetGauge("inferx_prompt_queued_bytes", static_cast<double>(health.queued_bytes));
  SetGauge("inferx_prompt_submitted_total",
           static_cast<double>(health.metrics.submitted));
  SetGauge("inferx_prompt_timed_out_total",
           static_cast<double>(health.metrics.timed_out));
}

std::string MetricsService::Render() const {
  std::lock_guard<std::mutex> lock(mu_);
  std::string out;
  out.reserve(4096);
  for (const auto& [name, value] : counters_) {
    out += name;
    out += ' ';
    out += std::to_string(value);
    out += '\n';
  }
  for (const auto& [name, value] : gauges_) {
    out += name;
    out += ' ';
    out += std::to_string(value);
    out += '\n';
  }
  return out;
}

}  // namespace inferx
