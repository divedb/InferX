#include "inferx/server/observability/metrics_snapshot.h"

#include "inferx/observe/metrics.h"

namespace inferx::server::observability {

std::string RenderPrometheus(const MetricsSnapshot& snapshot) {
  observe::Registry registry;
  auto gauge = [&](std::string name, std::string help, double value,
                   observe::Labels labels = {}) {
    registry.AddGauge(std::move(name), std::move(help), std::move(labels))
        ->Set(value);
  };
  auto counter = [&](std::string name, std::string help, uint64_t value) {
    registry.AddCounter(std::move(name), std::move(help))->Increment(value);
  };
  gauge("inferx_requests_running", "Requests currently executing.",
        snapshot.running);
  gauge("inferx_requests_waiting", "Requests waiting for admission.",
        snapshot.waiting);
  gauge("inferx_kv_blocks", "KV-cache blocks by state.",
        snapshot.blocks_in_use, {{"state", "used"}});
  gauge("inferx_kv_blocks", "KV-cache blocks by state.", snapshot.blocks_total,
        {{"state", "total"}});
  gauge("inferx_kv_blocks", "KV-cache blocks by state.", snapshot.cached_blocks,
        {{"state", "cached"}});
  const uint64_t free_blocks = snapshot.blocks_total > snapshot.blocks_in_use
                                   ? snapshot.blocks_total - snapshot.blocks_in_use
                                   : 0;
  gauge("inferx_kv_blocks", "KV-cache blocks by state.", free_blocks,
        {{"state", "free"}});
  gauge("inferx_kv_cache_usage_ratio", "Fraction of KV-cache blocks in use.",
        snapshot.blocks_total == 0
            ? 0.0
            : static_cast<double>(snapshot.blocks_in_use) /
                  snapshot.blocks_total);
  gauge("inferx_engine_last_step_seconds",
        "Device duration of the most recently completed engine step.",
        snapshot.last_step_ms / 1000.0);
  counter("inferx_steps_total", "Completed engine steps.", snapshot.steps);
  counter("inferx_preemptions_total", "Scheduler preemptions.",
          snapshot.preemptions);
  counter("inferx_prefix_cache_hits_total", "Prompt tokens served from cache.",
          snapshot.prefix_hit_tokens);
  counter("inferx_prefix_cache_misses_total",
          "Prompt tokens not served from cache.", snapshot.prefix_miss_tokens);
  counter("inferx_prefix_cache_evictions_total", "Evicted prefix-cache blocks.",
          snapshot.evicted_blocks);
  return registry.Render();
}

std::string RenderLegacyStatsJson(const MetricsSnapshot& snapshot) {
  return "{\"running\":" + std::to_string(snapshot.running) +
         ",\"waiting\":" + std::to_string(snapshot.waiting) +
         ",\"blocks_in_use\":" + std::to_string(snapshot.blocks_in_use) +
         ",\"blocks_total\":" + std::to_string(snapshot.blocks_total) +
         ",\"steps\":" + std::to_string(snapshot.steps) +
         ",\"tokens_generated\":" +
         std::to_string(snapshot.tokens_generated) + ",\"last_step_ms\":" +
         std::to_string(snapshot.last_step_ms) + ",\"preemptions\":" +
         std::to_string(snapshot.preemptions) + ",\"cached_blocks\":" +
         std::to_string(snapshot.cached_blocks) +
         ",\"prefix_hit_tokens\":" +
         std::to_string(snapshot.prefix_hit_tokens) +
         ",\"prefix_miss_tokens\":" +
         std::to_string(snapshot.prefix_miss_tokens) +
         ",\"evicted_blocks\":" + std::to_string(snapshot.evicted_blocks) +
         "}";
}

}  // namespace inferx::server::observability
