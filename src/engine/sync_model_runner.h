#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "inferx/core/kv_cache.h"
#include "inferx/core/status.h"
#include "inferx/model/forward_batch.h"

namespace inferx::model {
class DeepseekV2Model;
class GptOssModel;
}  // namespace inferx::model

namespace inferx::engine {

/// Type-erased contract for models whose step returns host-resident logits.
class SyncModelRunner {
 public:
  virtual ~SyncModelRunner() = default;

  virtual KvBlockPool* kv_pool() = 0;
  virtual int64_t vocab_size() const = 0;
  virtual Status ReserveActivations(int64_t max_tokens) = 0;
  virtual Status Step(const model::ForwardBatch& batch,
                      std::vector<float>* logits) = 0;
  virtual Status CaptureDecodeGraph(int64_t num_seqs,
                                    int64_t max_blocks_per_seq) = 0;
};

std::unique_ptr<SyncModelRunner> MakeSyncModelRunner(model::GptOssModel model);
std::unique_ptr<SyncModelRunner> MakeSyncModelRunner(
    model::DeepseekV2Model model);

}  // namespace inferx::engine
