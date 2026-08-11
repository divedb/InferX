#include "inferx/model/parallel/tp_dims.h"

namespace inferx::model::parallel {

StatusOr<TpDims> TpDims::For(const ModelConfig& config, const TpLayout& layout,
                             int tp_size) {
  INFERX_RETURN_IF_ERROR(layout.Validate(config, tp_size));

  TpDims dims;
  dims.local_heads = config.num_attention_heads / tp_size;
  dims.local_kv_heads = layout.kv_sharding() == KvSharding::kHeads
                            ? config.num_key_value_heads / tp_size
                            : config.num_key_value_heads;
  dims.local_q_dim = dims.local_heads * config.head_dim;
  dims.local_kv_dim = dims.local_kv_heads * config.head_dim;
  dims.local_intermediate = layout.Requires(TpDimension::kIntermediate)
                                ? config.intermediate_size / tp_size
                                : config.intermediate_size;
  dims.local_moe_intermediate = layout.Requires(TpDimension::kMoeIntermediate)
                                    ? config.moe_intermediate_size / tp_size
                                    : config.moe_intermediate_size;
  return dims;
}

}  // namespace inferx::model::parallel
