/// \file
/// \brief Parallel execution topology (vLLM ParallelConfig analogue).

#ifndef INFERX_CONFIG_PARALLEL_CONFIG_H_
#define INFERX_CONFIG_PARALLEL_CONFIG_H_

#include "inferx/core/status.h"

namespace inferx {

/// \brief How one engine instance shards a model across ranks.
///
/// Mirrors vLLM's ParallelConfig field names so the correspondence reads
/// directly; SGLang keeps the same values flat in ServerArgs. Only tensor
/// parallelism exists today, and only its geometry: collectives (all-reduce,
/// NCCL process groups) arrive with the tensor-parallel milestone. Rank-local
/// values derived from it -- per-rank attention heads, packed-projection row
/// shards -- flow through model build; the defaults make every derived value
/// identical to the unsharded model.
struct ParallelConfig {
  /// Ranks the model weights are sharded across; one worker process per
  /// device when greater than 1.
  /// EXAMPLE: --tensor-parallel-size 2
  int tensor_parallel_size = 1;

  /// This instance's rank, in [0, tensor_parallel_size); the controller
  /// assigns each spawned worker its rank, so it is never set from the CLI.
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
