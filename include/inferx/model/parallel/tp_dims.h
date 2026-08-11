#pragma once

#include <cstdint>

#include "inferx/core/status.h"
#include "inferx/model/config.h"
#include "inferx/model/parallel/tp_layout.h"

namespace inferx::model::parallel {

struct TpDims {
  static StatusOr<TpDims> For(const ModelConfig& config, const TpLayout& layout,
                              int tp_size);

  int64_t local_heads = 0;
  int64_t local_kv_heads = 0;
  int64_t local_q_dim = 0;
  int64_t local_kv_dim = 0;
  int64_t local_intermediate = 0;
  int64_t local_moe_intermediate = 0;
};

}  // namespace inferx::model::parallel
