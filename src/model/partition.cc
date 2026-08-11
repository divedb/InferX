#include "inferx/model/parallel/partition.h"

namespace inferx::model::parallel {

StatusOr<int> ResolveShardAxis(const ShardSpec& shard, int tensor_rank) {
  if (tensor_rank < 0) {
    return InvalidArgumentError("tensor rank must not be negative");
  }
  if (shard.partition == Partition::kReplicated) return -1;

  int axis = shard.axis;
  if (axis == -1) {
    switch (shard.partition) {
      case Partition::kRows:
      case Partition::kVocab:
        axis = 0;
        break;
      case Partition::kCols:
        axis = 1;
        break;
      case Partition::kReplicated:
        return -1;
    }
  } else if (axis < 0) {
    axis += tensor_rank;
  }
  if (axis < 0 || axis >= tensor_rank) {
    return InvalidArgumentError("shard axis ", shard.axis,
                                " is outside tensor rank ", tensor_rank);
  }
  return axis;
}

}  // namespace inferx::model::parallel
