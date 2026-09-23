#ifndef INFERX_BENCH_THROUGHPUT_H_
#define INFERX_BENCH_THROUGHPUT_H_

#include <string>

#include "inferx/bench/dataset.h"
#include "inferx/core/status.h"
#include "inferx/cache/cache_config.h"
#include "inferx/engine/execution_config.h"
#include "inferx/engine/scheduler.h"
#include "inferx/models/model_config.h"

namespace inferx::bench {

/// \brief Inputs for the offline-throughput benchmark
///        (`inferx bench throughput`). The CLI-independent contract: tests,
///        harnesses, and future front ends construct it directly.
struct ThroughputParams {
  ModelConfig model;
  CacheConfig cache;
  SchedulerConfig scheduler;
  ExecutionConfig execution;
  DatasetParams dataset;
  int num_prompts = 64;
  int num_iters = 3;         ///< Timed trials after warmup (workload default).
  int num_iters_warmup = 1;  ///< Untimed warmup trials.
  std::string output_json;         ///< Results JSON path; empty is stdout only.
  bool disable_log_stats = false;  ///< Suppress statistics logging.
};

/// \brief Runs the benchmark, printing one JSON record per (case, repeat) to
///        stdout.
Status RunThroughput(const ThroughputParams& params);

}  // namespace inferx::bench

#endif  // INFERX_BENCH_THROUGHPUT_H_
