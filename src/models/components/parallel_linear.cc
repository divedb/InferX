#include "inferx/models/components/parallel_linear.h"

namespace inferx {
namespace components {

StatusOr<DimShard> ShardDim(int64_t total, const ParallelConfig& parallel) {
  INFERX_RETURN_IF_ERROR(parallel.Validate());
  if (total <= 0) {
    return InvalidArgumentError("cannot shard non-positive extent ", total);
  }
  if (total % parallel.tensor_parallel_size != 0) {
    return InvalidArgumentError("extent ", total, " does not divide tensor parallel size ",
                                parallel.tensor_parallel_size);
  }
  const int64_t size = total / parallel.tensor_parallel_size;
  return DimShard{size, parallel.tensor_parallel_rank * size};
}

}  // namespace inferx::components
}  // namespace inferx
