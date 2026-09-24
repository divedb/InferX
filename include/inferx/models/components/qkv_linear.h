/// \file
/// \brief QKVParallelLinear: the fused, head-sharded QKV projection
///        (vLLM QKVParallelLinear analogue).

#ifndef INFERX_MODELS_COMPONENTS_QKV_LINEAR_H_
#define INFERX_MODELS_COMPONENTS_QKV_LINEAR_H_

#include <cstdint>

#include "inferx/core/status.h"
#include "inferx/engine/parallel_config.h"
#include "inferx/models/components/attention.h"

namespace inferx {
namespace components {

/// \brief One rank's slice of the fused QKV projection.
///
/// All fields are rank-local except the replication bookkeeping. Row counts
/// describe the packed weight this rank loads and executes:
/// [query_rows | kv_rows | kv_rows] x hidden, block-contiguous.
struct QkvParallelGeometry {
  int64_t query_heads = 0;        ///< Query heads on this rank.
  int64_t kv_heads = 0;           ///< Key/value heads on this rank; at least 1.
  int64_t head_dim = 0;
  int64_t query_rows = 0;         ///< query_heads * head_dim (* 2 when gated).
  int64_t kv_rows = 0;            ///< kv_heads * head_dim.
  int64_t kv_head_replicas = 1;   ///< Ranks sharing each KV head when total < size.
  int64_t kv_shard = 0;           ///< Which KV-head shard this rank loads.
};

/// \brief Derives one rank's geometry from the total attention configuration.
///
/// Query heads divide across ranks (total must be divisible). Key/value heads
/// divide while total >= tensor_parallel_size; below it each rank takes one
/// KV head and `kv_head_replicas` ranks replicate it (vLLM QKVParallelLinear
/// semantics). A gated output doubles the query rows this rank loads.
StatusOr<QkvParallelGeometry> ShardQkv(const AttentionConfig& total,
                                       const ParallelConfig& parallel);

}  // namespace inferx::components
}  // namespace inferx

#endif  // INFERX_MODELS_COMPONENTS_QKV_LINEAR_H_
