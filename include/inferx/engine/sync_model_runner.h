#pragma once

#include <memory>

#include "inferx/engine/host_sampling.h"
#include "inferx/engine/model_runner.h"

namespace inferx::model {
class DeepseekV2Model;
class GptOssModel;
}  // namespace inferx::model

namespace inferx::engine {

std::unique_ptr<ModelRunner> MakeSyncModelRunner(model::GptOssModel model,
                                                 HostSampler sampler);
std::unique_ptr<ModelRunner> MakeSyncModelRunner(model::DeepseekV2Model model,
                                                 HostSampler sampler);

}  // namespace inferx::engine
