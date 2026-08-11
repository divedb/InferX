#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>

#include "inferx/model/forward_batch.h"
#include "model_runner.h"

namespace inferx::engine {

using HostSampler = std::function<int32_t(
    float* row, int64_t vocab, const model::ForwardBatch& batch, size_t index,
    SampledLogprob* logprob)>;

// Samples one host-resident logits row for synchronous model runners.
int32_t HostSampleRow(float* row, int64_t vocab,
                      const model::ForwardBatch& batch, size_t index,
                      SampledLogprob* logprob_out);

}  // namespace inferx::engine
