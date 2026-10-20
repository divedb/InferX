#ifndef INFERX_CONFIG_LORA_CONFIG_H_
#define INFERX_CONFIG_LORA_CONFIG_H_

#include <string>

#include "inferx/core/status.h"

namespace inferx {

/// \brief LoRA adapter serving (vLLM: --enable-lora, --max-loras, ...).
struct LoRAConfig {
  bool enable_lora = false;
  /// Largest adapter rank that can be loaded. One of 1, 8, 16, 32, 64, 128,
  /// 256, 320, 512.
  int max_lora_rank = 16;
  /// Adapters resident in one batch (GPU slots).
  int max_loras = 1;
  /// Adapters cached in CPU memory. 0 means "same as max_loras".
  int max_cpu_loras = 0;
  /// Adapter dtype: "auto" follows the base model, or "float16"/"bfloat16".
  std::string lora_dtype = "auto";
  /// Shard LoRA layers across tensor-parallel ranks as well.
  bool fully_sharded_loras = false;

  /// Effective CPU cache size after applying the 0 -> max_loras default.
  int EffectiveMaxCpuLoras() const { return max_cpu_loras == 0 ? max_loras : max_cpu_loras; }

  /// Checks the fields only when `enable_lora` is set.
  Status Validate() const;
};

}  // namespace inferx

#endif  // INFERX_CONFIG_LORA_CONFIG_H_