#pragma once

#include <memory>

#include "inferx/core/device.h"
#include "inferx/core/status.h"
#include "inferx/model/config.h"
#include "host_sampling.h"
#include "model_runner.h"

namespace inferx::engine {

struct EngineConfig;

// Builds the architecture-specific model and adapts it to the common serving
// contract. Model loading, quantization validation, and KV allocation belong
// together here; scheduling and request lifecycle remain Engine concerns.
StatusOr<std::unique_ptr<ModelRunner>> BuildModelRunner(
    const EngineConfig& config, model::Architecture architecture,
    DeviceId primary_device, HostSampler host_sampler);

}  // namespace inferx::engine
