#pragma once

#include <vector>

#include "inferx/core/status.h"
#include "inferx/core/tensor.h"

namespace inferx::comm {

/// Copies rank `rank`'s equal contiguous shard of a CPU tensor along `axis`.
/// Tensor-parallel checkpoint weights use axis 0 for column-parallel output
/// features and axis 1 for row-parallel input features.
StatusOr<Tensor> ShardHostTensor(const Tensor& tensor, int axis, int rank,
                                 int world_size);

/// Inverse of ShardHostTensor, used by M7 reconstruction tests.
StatusOr<Tensor> ReconstructHostTensor(const std::vector<Tensor>& shards,
                                       int axis);

}  // namespace inferx::comm
