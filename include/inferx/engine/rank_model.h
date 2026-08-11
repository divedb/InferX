#pragma once

#include <cstdint>
#include <vector>

#include "inferx/core/kv_cache.h"
#include "inferx/core/status.h"
#include "inferx/engine/model_runner.h"
#include "inferx/model/forward_batch.h"

namespace inferx::engine {

// Type-erased model surface executed by one tensor-parallel rank. Architecture
// adapters own model-specific loading and preparation; rank workers own only
// execution, failure propagation, and topology.
class RankModel {
 public:
  virtual ~RankModel() = default;

  virtual Status PrepareWeights() = 0;
  virtual int64_t KvBlockBytes(int64_t block_size) const = 0;
  virtual Status AttachKvCache(int64_t blocks, int64_t block_size) = 0;
  virtual Status EnableSampling(int64_t max_rows) = 0;
  virtual KvBlockPool* kv_pool() = 0;

  virtual Status ReserveActivations(int64_t max_tokens) = 0;
  virtual Status StepAsync(const model::ForwardBatch& batch) = 0;
  virtual Status AwaitStep(std::vector<int32_t>* sampled) = 0;
  virtual Status ReadSampledLogprobs(std::vector<SampledLogprob>* logprobs) = 0;
  virtual Status CaptureDecodeGraph(int64_t num_seqs,
                                    int64_t max_blocks_per_seq) = 0;
  virtual Status AbortCommunication() = 0;
  virtual float last_step_device_ms() const = 0;
};

}  // namespace inferx::engine
