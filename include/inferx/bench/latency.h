#ifndef INFERX_BENCH_LATENCY_H_
#define INFERX_BENCH_LATENCY_H_

#include <string>

#include "inferx/core/status.h"
#include "inferx/cache/cache_config.h"
#include "inferx/engine/execution_config.h"
#include "inferx/engine/scheduler.h"
#include "inferx/models/model_config.h"

namespace inferx::bench {

/// \brief Inputs for the batch-latency benchmark (`inferx bench latency`).
struct LatencyParams {
  ModelConfig model;
  CacheConfig cache;
  SchedulerConfig scheduler;
  ExecutionConfig execution;
  int batch_size = 8;    ///< Requests submitted as one batch.
  int output_len = 256;  ///< Tokens generated per request.
  int input_len = 32;         ///< Length of the synthetic prompts.
  int num_iters = 30;         ///< Timed iterations (vLLM default).
  int num_iters_warmup = 10;  ///< Untimed warmup iterations (vLLM default).
  std::string output_json;         ///< Results JSON path; empty is stdout only.
  bool disable_log_stats = false;  ///< Suppress statistics logging.
};

/// \brief Runs the benchmark, printing one JSON record per repeat to stdout.
Status RunLatency(const LatencyParams& params);

}  // namespace inferx::bench

#endif  // INFERX_BENCH_LATENCY_H_
