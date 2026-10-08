/// \file
/// \brief Linear projections and the shard math for the plain
///        parallel-linear family (MergedColumnParallelLinear,
///        RowParallelLinear, VocabParallelEmbedding analogues).

#ifndef INFERX_MODELS_LAYERS_LINEAR_H_
#define INFERX_MODELS_LAYERS_LINEAR_H_

#include <cstdint>
#include <optional>

#include "inferx/config/parallel_config.h"
#include "inferx/core/status.h"
#include "inferx/core/tensor.h"

namespace inferx::models::layers {

/// \brief A projection's weight and optional bias.
struct LinearWeights {
  Tensor weight;                    ///< [out, in]
  std::optional<Tensor> bias;       ///< [out]; absent when bias-free
};

/// \brief One rank's slice of a dimension sharded across ranks.
struct DimShard {
  int64_t size = 0;   ///< Extent on this rank.
  int64_t begin = 0;  ///< Offset of this rank's slice in the full extent.
};

/// \brief Divides `total` evenly across the tensor-parallel ranks and
///        returns this rank's slice. `total` must divide
///        `tensor_parallel_size`; vocabulary padding for non-divisible
///        checkpoints arrives with the tensor-parallel milestone.
StatusOr<DimShard> ShardDim(int64_t total, const ParallelConfig& parallel);

}  // namespace inferx::models::layers

#endif  // INFERX_MODELS_LAYERS_LINEAR_H_
