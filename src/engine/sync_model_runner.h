#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

#include "model_runner.h"

namespace inferx::model {
class DeepseekV2Model;
class GptOssModel;
}  // namespace inferx::model

namespace inferx::engine {

using HostSampler = std::function<int32_t(
    float* row, int64_t vocab, const model::ForwardBatch& batch, size_t index,
    SampledLogprob* logprob)>;

std::unique_ptr<ModelRunner> MakeSyncModelRunner(model::GptOssModel model,
                                                 HostSampler sampler);
std::unique_ptr<ModelRunner> MakeSyncModelRunner(model::DeepseekV2Model model,
                                                 HostSampler sampler);

}  // namespace inferx::engine
