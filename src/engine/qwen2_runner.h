#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "inferx/core/status.h"
#include "model_runner.h"

namespace inferx::engine {

struct Qwen2RunnerConfig {
  std::string model_dir;
  DeviceKind device_kind = DeviceKind::kCuda;
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

/// Model execution boundary used by the server. TP=1 calls the model directly;
/// TP=2 dispatches the same operation to two persistent rank workers.
class Qwen2Runner : public ModelRunner {
 public:
  ~Qwen2Runner() override = default;

  static StatusOr<std::unique_ptr<Qwen2Runner>> Create(
      const Qwen2RunnerConfig& config);

  bool SupportsGraphCapture() const override { return true; }
  bool RequiresGraphWarmup() const override { return true; }
};

}  // namespace inferx::engine
