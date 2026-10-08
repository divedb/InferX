/// \file
/// \brief Llama dense configuration: the canonical pre-norm decoder
/// layout (no Q/K norm, no attention biases, llama3-style rope defaults
/// come from the checkpoint).

#ifndef INFERX_MODELS_ARCHITECTURES_LLAMA_CONFIG_H_
#define INFERX_MODELS_ARCHITECTURES_LLAMA_CONFIG_H_

#include "inferx/core/status.h"
#include "inferx/models/configs/dims.h"
#include "inferx/models/configs/rope.h"
#include "inferx/util/config_view.h"

namespace inferx::models::llama {

struct LlamaConfig : ModelDims {
  RopeConfig rope;

  static StatusOr<LlamaConfig> Parse(const ConfigView& view);
  Status Validate() const;
};

}  // namespace inferx::models::llama

#endif  // INFERX_MODELS_ARCHITECTURES_LLAMA_CONFIG_H_
