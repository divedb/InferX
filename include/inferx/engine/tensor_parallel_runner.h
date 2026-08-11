#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

#include "inferx/core/status.h"
#include "inferx/engine/model_runner.h"
#include "inferx/engine/parallel_context.h"
#include "inferx/engine/rank_model.h"

namespace inferx::engine {

struct TensorParallelRunnerConfig {
  DeviceKind device_kind = DeviceKind::kCuda;
  std::vector<int> devices{0};
  bool use_nccl = false;
  uint64_t collective_timing_sample_every = 0;
  int64_t kv_blocks = 0;
  int64_t kv_cache_memory_bytes = 0;
  double gpu_memory_utilization = 0.92;
  int64_t block_size = 16;
  int64_t max_seq_len = 2048;
  int64_t max_sampling_rows = 8;
  bool supports_graph_capture = false;
  bool requires_graph_warmup = false;
};

using RankModelFactory = std::function<StatusOr<std::unique_ptr<RankModel>>(
    ParallelContext context)>;

// Architecture-neutral TP execution mechanism. Model factories supply one
// rank-local strategy; this class supplies workers, collectives, rendezvous,
// failure propagation, and rank-zero result selection.
class TensorParallelRunner : public ModelRunner {
 public:
  ~TensorParallelRunner() override = default;

  static StatusOr<std::unique_ptr<TensorParallelRunner>> Create(
      const TensorParallelRunnerConfig& config, RankModelFactory factory);

  bool SupportsGraphCapture() const override { return supports_graph_capture_; }
  bool RequiresGraphWarmup() const override { return requires_graph_warmup_; }

 protected:
  explicit TensorParallelRunner(const TensorParallelRunnerConfig& config)
      : supports_graph_capture_(config.supports_graph_capture),
        requires_graph_warmup_(config.requires_graph_warmup) {}

 private:
  bool supports_graph_capture_ = false;
  bool requires_graph_warmup_ = false;
};

}  // namespace inferx::engine
