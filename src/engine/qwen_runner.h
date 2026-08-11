#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "inferx/comm/communicator.h"
#include "inferx/core/kv_cache.h"
#include "inferx/core/status.h"
#include "inferx/model/qwen2.h"

namespace inferx::engine {

struct QwenRunnerConfig {
  std::string model_dir;
  std::vector<int> devices{0};
  bool use_nccl = false;
  uint64_t collective_timing_sample_every = 0;
  bool fp8_weights = false;
  bool int4_weights = false;
  bool fp8_kv_cache = false;
  /// KV pool sizing: an explicit block count pins the pool; zero auto-sizes
  /// from `kv_cache_memory_bytes` or `gpu_memory_utilization` after weights
  /// load, with TP ranks agreeing on the minimum across devices.
  int64_t kv_blocks = 0;
  int64_t kv_cache_memory_bytes = 0;
  double gpu_memory_utilization = 0.92;
  int64_t block_size = 16;
  /// Longest sequence; auto-sizing refuses a pool too small to hold one.
  int64_t max_seq_len = 2048;
  int64_t max_sampling_rows = 8;
};

struct RankTelemetry {
  int rank = 0;
  int device = 0;
  int world_size = 1;
  comm::CommBackend backend = comm::CommBackend::kSingleRank;
  bool healthy = true;
  double last_progress_age_seconds = 0.0;
  float last_step_device_ms = 0.0f;
  uint64_t timeouts = 0;
  comm::CommMetricSnapshot communication;
};

/// Model execution boundary used by the server. TP=1 calls the model directly;
/// TP=2 dispatches the same operation to two persistent rank workers.
class QwenRunner {
 public:
  virtual ~QwenRunner() = default;

  static StatusOr<std::unique_ptr<QwenRunner>> Create(
      const QwenRunnerConfig& config);

  virtual KvBlockPool* kv_pool() = 0;
  virtual Status ReserveActivations(int64_t max_tokens) = 0;
  virtual Status StepAsync(const model::ForwardBatch& batch) = 0;
  virtual Status AwaitStep(std::vector<int32_t>* sampled) = 0;
  /// Valid after `AwaitStep`; rank 0's readback (all ranks sample
  /// identically). \see Qwen2Model::ReadSampledLogprobs.
  virtual Status ReadSampledLogprobs(
      std::vector<model::Qwen2Model::SampledLogprob>* out) = 0;
  virtual Status CaptureDecodeGraph(int64_t num_seqs,
                                    int64_t max_blocks_per_seq) = 0;
  virtual float last_step_device_ms() const = 0;
  virtual std::vector<RankTelemetry> telemetry() const = 0;
};

}  // namespace inferx::engine
