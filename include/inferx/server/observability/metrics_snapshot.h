#pragma once

#include <cstdint>
#include <string>

namespace inferx::server::observability {

struct MetricsSnapshot {
  uint64_t running = 0;
  uint64_t waiting = 0;
  uint64_t blocks_in_use = 0;
  uint64_t blocks_total = 0;
  uint64_t steps = 0;
  uint64_t tokens_generated = 0;
  double last_step_ms = 0;
  uint64_t preemptions = 0;
  uint64_t cached_blocks = 0;
  uint64_t prefix_hit_tokens = 0;
  uint64_t prefix_miss_tokens = 0;
  uint64_t evicted_blocks = 0;
};

class MetricsSource {
 public:
  virtual ~MetricsSource() = default;
  virtual MetricsSnapshot Snapshot() const = 0;
};

std::string RenderPrometheus(const MetricsSnapshot& snapshot);
std::string RenderLegacyStatsJson(const MetricsSnapshot& snapshot);

}  // namespace inferx::server::observability
