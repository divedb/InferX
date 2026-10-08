#ifndef INFERX_CONFIG_PARALLEL_CONFIG_H_
#define INFERX_CONFIG_PARALLEL_CONFIG_H_

#include "inferx/core/status.h"

namespace inferx {

/// \brief How one engine instance shards a model across ranks.
///
/// The model loader shards weights and persistent state (KV cache, etc.) using
/// this geometry. Ranks are laid out on a (data, pipeline, tensor) grid, so the
/// total number of ranks is `WorldSize() = dp * pp * tp`.
///
/// Multi-GPU serving drives one rank per distinct CUDA device in one process,
/// with one scheduler and NCCL collectives. Explicit rank-local callers supply
/// a matching CommBackend to the model runner.
///
/// Only the sizes and ranks are stored here. Checks that need the model config
/// (heads divisible by TP, layers by PP, experts by EP, KV-head replication
/// under GQA/MQA) are the loader's responsibility.
struct ParallelConfig {
  /// \name Group sizes
  /// @{

  /// Tensor parallelism: ranks that each hold a slice of every layer's weights
  /// (attention heads, MLP columns/rows). Layers finish with an all-reduce
  /// across the group.
  /// EXAMPLE: --tensor-parallel-size 2
  int tensor_parallel_size = 1;

  /// Pipeline parallelism: ranks that each own a contiguous range of layers.
  /// Activations are passed between consecutive stages point-to-point.
  /// EXAMPLE: --pipeline-parallel-size 2
  int pipeline_parallel_size = 1;

  /// Data parallelism: independent replicas that serve different requests, so
  /// each replica holds its own KV cache. For MoE models this is used as
  /// attention DP, where the attention layers run per replica and the expert
  /// layers are shared through expert parallelism.
  /// EXAMPLE: --data-parallel-size 2
  int data_parallel_size = 1;

  /// Expert parallelism: place whole MoE experts on different ranks instead of
  /// sharding each expert with TP. Tokens are routed with all-to-all
  /// dispatch/combine. This is not an independent axis: the EP group reuses the
  /// existing `dp * tp` ranks within each pipeline stage (see
  /// ExpertParallelSize()). Has no effect on dense models.
  /// EXAMPLE: --enable-expert-parallel
  bool enable_expert_parallel = false;

  /// @}

  /// \name This instance's coordinates
  /// Each rank is in [0, corresponding size).
  /// @{

  /// Position in the tensor-parallel group.
  int tensor_parallel_rank = 0;

  /// Pipeline stage index. Stage 0 owns the embedding; the last stage owns the
  /// final norm and LM head.
  int pipeline_parallel_rank = 0;

  /// Index of the data-parallel replica.
  int data_parallel_rank = 0;

  /// @}

  /// \brief Total ranks (one CUDA device each in multi-GPU serving).
  int WorldSize() const {
    return tensor_parallel_size * pipeline_parallel_size * data_parallel_size;
  }

  /// \brief Size of the expert-parallel group; 1 when EP is disabled.
  ///
  /// Derived rather than configurable so callers can't supply a value that
  /// is inconsistent with TP and DP.
  ///
  /// \return The number of ranks that share each expert (the EP group size).
  int ExpertParallelSize() const {
    return enable_expert_parallel ? tensor_parallel_size * data_parallel_size : 1;
  }

  /// \brief Checks that sizes are positive and each rank is within its group.
  ///
  /// Does not validate against any model config.
  ///
  /// \return OK if the config is valid, or InvalidArgumentError naming the
  ///         first problem.
  Status Validate() const {
    if (tensor_parallel_size <= 0 || pipeline_parallel_size <= 0 || data_parallel_size <= 0) {
      return InvalidArgumentError("parallel sizes must be positive");
    }

    if (tensor_parallel_rank < 0 || tensor_parallel_rank >= tensor_parallel_size ||
        pipeline_parallel_rank < 0 || pipeline_parallel_rank >= pipeline_parallel_size ||
        data_parallel_rank < 0 || data_parallel_rank >= data_parallel_size) {
      return InvalidArgumentError("rank out of range for its group size");
    }

    return OkStatus();
  }
};

}  // namespace inferx

#endif  // INFERX_CONFIG_PARALLEL_CONFIG_H_
