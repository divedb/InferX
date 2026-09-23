#ifndef INFERX_BENCH_WORKLOAD_H_
#define INFERX_BENCH_WORKLOAD_H_

#include <string>

#include "inferx/core/status.h"
#include "inferx/cache/cache_config.h"
#include "inferx/engine/execution_config.h"
#include "inferx/engine/scheduler.h"
#include "inferx/models/model_config.h"

namespace inferx::bench {

/// \brief Inputs for the fixed-suite benchmark (`inferx bench workload`, an
///        InferX extension; not part of vLLM's command tree).
///
/// Engine capacity (pool size, token budget, batch) arrives via
/// runner/scheduler. The suite JSON contributes workload data (cases,
/// prompts, output lengths) and, when present, overrides token_budget,
/// kv_blocks, and block_size so existing suites keep pinning them.
struct WorkloadParams {
  ModelConfig model;
  CacheConfig cache;
  SchedulerConfig scheduler;
  ExecutionConfig execution;
  std::string suite = "benchmarks/qwen3/workload.json";
  int repeats = 3;         ///< Timed trials per case after one warmup.
  int profile_step = -1;   ///< CUDA profiler range during repeat 0; -1 disables.
  bool step_timings = false;  ///< Emit per-step timing CSV to stderr.
  std::string output_json;         ///< Results JSON path; empty is stdout only.
  bool disable_log_stats = false;  ///< Suppress statistics logging.
};

/// \brief Runs the suite through the real scheduler and model runner,
///        printing one JSON record per (case, repeat) to stdout.
Status RunWorkload(const WorkloadParams& params);

}  // namespace inferx::bench

#endif  // INFERX_BENCH_WORKLOAD_H_
