#pragma once

#include "inferx/core/status.h"
#include "inferx/model/config.h"

namespace inferx::engine {

struct ModelRunnerFeatures {
  int tensor_parallel_size = 1;
  bool fp8_weights = false;
  bool int4_weights = false;
  bool fp8_kv_cache = false;
};

// Validates architecture capability requests without loading a checkpoint or
// touching a device, so startup policy remains independently testable.
inline Status ValidateModelRunnerFeatures(model::Architecture architecture,
                                          const ModelRunnerFeatures& features) {
  const bool synchronous = architecture == model::Architecture::kDeepSeekV2 ||
                           architecture == model::Architecture::kGptOss;
  if (!synchronous) return OkStatus();

  if (features.tensor_parallel_size != 1) {
    return UnimplementedError(
        "tensor parallel serving is currently implemented for Qwen2 only");
  }
  if (features.fp8_weights || features.int4_weights ||
      features.fp8_kv_cache) {
    return InvalidArgumentError(
        "--quantization / --kv-cache-dtype are not supported for the ",
        model::ArchitectureName(architecture), " architecture");
  }
  return OkStatus();
}

}  // namespace inferx::engine
