#include "inferx/config/lora_config.h"

#include <algorithm>
#include <array>
#include <string>

namespace inferx {

namespace {

constexpr std::array<int, 9> kSupportedRanks = {1, 8, 16, 32, 64, 128, 256, 320, 512};

}  // namespace

Status LoRAConfig::Validate() const {
  if (!enable_lora) return Status::OK();

  if (std::find(kSupportedRanks.begin(), kSupportedRanks.end(), max_lora_rank) ==
      kSupportedRanks.end()) {
    return Status::InvalidArgument(
        "LoRAConfig: max_lora_rank must be one of 1, 8, 16, 32, 64, 128, 256, 320, 512, got " +
        std::to_string(max_lora_rank));
  }
  if (max_loras < 1) {
    return Status::InvalidArgument("LoRAConfig: max_loras must be >= 1");
  }
  if (max_cpu_loras < 0) {
    return Status::InvalidArgument("LoRAConfig: max_cpu_loras must be >= 0");
  }
  if (EffectiveMaxCpuLoras() < max_loras) {
    return Status::InvalidArgument(
        "LoRAConfig: max_cpu_loras (" + std::to_string(EffectiveMaxCpuLoras()) +
        ") must be >= max_loras (" + std::to_string(max_loras) + ")");
  }
  if (lora_dtype != "auto" && lora_dtype != "float16" && lora_dtype != "bfloat16") {
    return Status::InvalidArgument(
        "LoRAConfig: lora_dtype must be one of auto, float16, "
        "bfloat16, got '" +
        lora_dtype + "'");
  }
  return Status::OK();
}

}  // namespace inferx