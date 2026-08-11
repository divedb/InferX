#pragma once

#include <cstdint>
#include <utility>
#include <vector>

#include "inferx/comm/communicator.h"
#include "inferx/core/kv_cache.h"
#include "inferx/core/status.h"
#include "inferx/model/forward_batch.h"

namespace inferx::engine {

struct SampledLogprob {
  bool present = false;
  float logprob = 0.0f;
  std::vector<std::pair<int32_t, float>> top;
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

/// Backend- and architecture-neutral asynchronous model execution contract.
/// Architecture adapters own loading and any synchronous/device-specific
/// details needed to implement it.
class ModelRunner {
 public:
  virtual ~ModelRunner() = default;

  virtual KvBlockPool* kv_pool() = 0;
  virtual Status ReserveActivations(int64_t max_tokens) = 0;
  virtual Status Step(const model::ForwardBatch& batch,
                      std::vector<int32_t>* sampled,
                      std::vector<SampledLogprob>* logprobs) = 0;
  virtual Status CaptureDecodeGraph(int64_t num_seqs,
                                    int64_t max_blocks_per_seq) = 0;
  virtual float last_step_device_ms() const = 0;
  virtual std::vector<RankTelemetry> telemetry() const = 0;
};

}  // namespace inferx::engine
