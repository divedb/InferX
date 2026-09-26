/// \file
/// \brief Shard math for the plain parallel-linear family (vLLM
///        MergedColumnParallelLinear, RowParallelLinear, and
///        VocabParallelEmbedding analogues).
///
/// Unlike QKVParallelLinear (qkv_linear.h), whose shards follow attention
/// head geometry with KV replication, these layers shard one dimension
/// evenly: MergedColumnParallelLinear the gate/up output rows,
/// RowParallelLinear the input columns, VocabParallelEmbedding the vocab
/// rows. One division rule serves all three.
///
/// Execution note: RowParallelLinear partial sums and VocabParallel logits
/// are only complete after cross-rank collectives, which arrive with the
/// tensor-parallel milestone; at tensor_parallel_size == 1 every shard is
/// the whole tensor and execution is unchanged.

#ifndef INFERX_MODELS_COMPONENTS_PARALLEL_LINEAR_H_
#define INFERX_MODELS_COMPONENTS_PARALLEL_LINEAR_H_

#include <cstdint>

#include "inferx/core/status.h"
#include "inferx/config/parallel_config.h"

namespace inferx {
namespace components {

/// \brief One rank's slice of a dimension sharded across ranks.
struct DimShard {
  int64_t size = 0;   ///< Extent on this rank.
  int64_t begin = 0;  ///< Offset of this rank's slice in the full extent.
};

/// \brief Divides `total` evenly across the tensor-parallel ranks and
///        returns this rank's slice.
///
/// `total` must divide `tensor_parallel_size`; vocabulary padding for
/// non-divisible checkpoints arrives with the tensor-parallel milestone.
StatusOr<DimShard> ShardDim(int64_t total, const ParallelConfig& parallel);

}  // namespace inferx::components
}  // namespace inferx

#endif  // INFERX_MODELS_COMPONENTS_PARALLEL_LINEAR_H_
