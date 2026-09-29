#ifndef INFERX_CONFIG_PARALLEL_CONFIG_H_
#define INFERX_CONFIG_PARALLEL_CONFIG_H_

#include "inferx/core/status.h"

namespace inferx {

/// \brief How one engine instance shards a model across ranks.
///
/// The model loader shards weights and persistent state using this geometry.
/// Multi-GPU serving drives one rank per distinct CUDA device in one process,
/// with one scheduler and NCCL collectives. Explicit rank-local callers supply
/// a matching CommBackend to the model runner.
struct ParallelConfig {
  /// Ranks the model weights are sharded across; one CUDA device per rank
  /// in multi-GPU serving.
  /// EXAMPLE: --tensor-parallel-size 2
  int tensor_parallel_size = 1;

  /// This instance's rank, in [0, tensor_parallel_size). The serving coordinator
  /// starts at rank zero and assigns ranks to its GPU execution lanes.
  int tensor_parallel_rank = 0;

  /// \brief Checks rank/size sanity.
  Status Validate() const {
    if (tensor_parallel_size <= 0 || tensor_parallel_rank < 0 ||
        tensor_parallel_rank >= tensor_parallel_size) {
      return InvalidArgumentError("invalid tensor-parallel topology: size ",
                                  tensor_parallel_size, ", rank ", tensor_parallel_rank);
    }

    return OkStatus();
  }
};

}  // namespace inferx

#endif  // INFERX_CONFIG_PARALLEL_CONFIG_H_
