/// \brief Low-cardinality metrics service with Prometheus text rendering
/// (docs/inference_server_design.md §7.6/§12.5).
///
/// Counters and gauges keyed by pre-joined label strings; callers pass
/// bounded label sets only (route/method/status, reason categories) —
/// never request IDs, prompts, or raw client strings. Exposed at
/// GET /metrics.
#ifndef INFERX_SERVER_METRICS_H_
#define INFERX_SERVER_METRICS_H_

#include <cstdint>
#include <map>
#include <string>
#include <string_view>
#include <vector>

#include "inferx/server/prompt_processor_pool.h"

namespace inferx {

class MetricsService {
 public:
  /// name carries the full label set, e.g.
  /// `http_requests_total{route="/v1/completions",status="200"}`.
  void Increment(std::string_view name, std::uint64_t by = 1);
  void SetGauge(std::string_view name, double value);

  /// Convenience helpers for the fixed label sets the server uses.
  void ObserveRequest(const std::string& route, const std::string& method, int status);
  void ObservePromptRejected(std::string_view reason);
  void ObserveDisconnect();
  void ObserveEngineAbort();

  /// Snapshots prompt-pool gauges from a health value.
  void ObservePoolHealth(const PoolHealth& health);

  /// Prometheus text exposition (0.0.4): counters, then gauges, ordered.
  std::string Render() const;

 private:
  mutable std::mutex mu_;
  std::map<std::string, std::uint64_t> counters_;
  std::map<std::string, double> gauges_;
};

}  // namespace inferx

#endif  // INFERX_SERVER_METRICS_H_
