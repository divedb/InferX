#ifndef INFERX_MODELS_ARCHITECTURES_QWEN2_CONFIG_H_
#define INFERX_MODELS_ARCHITECTURES_QWEN2_CONFIG_H_

#include "inferx/core/status.h"
#include "inferx/models/configs/dims.h"
#include "inferx/models/configs/rope.h"
#include "inferx/util/config_view.h"

namespace inferx::models::qwen2 {

struct Qwen2Config : ModelDims {
  RopeConfig rope;

  static StatusOr<Qwen2Config> Parse(const ConfigView& view);
  Status Validate() const;
};

}  // namespace inferx::models::qwen2

#endif  // INFERX_MODELS_ARCHITECTURES_QWEN2_CONFIG_H_
