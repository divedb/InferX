#ifndef INFERX_CONFIG_PARALLEL_CONFIG_H_
#define INFERX_CONFIG_PARALLEL_CONFIG_H_

#include <array>
#include <chrono>
#include <cstdint>
#include <string>
#include <vector>

#include "inferx/core/status.h"

namespace inferx {

enum class Axis : std::uint8_t {
  kDataParallel = 0,
  kPipelineParallel = 1,
  kContextParallel = 2,
  kTensorParallel = 3,
  kAxisCount,
};

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

  /// Data parallelism: independent replicas that serve different requests, so
  /// each replica holds its own KV cache. For MoE models this is used as
  /// attention DP, where the attention layers run per replica and the expert
  /// layers are shared through expert parallelism.
  /// EXAMPLE: --data-parallel-size 2
  int data_parallel_size = 1;

  /// Pipeline parallelism: ranks that each own a contiguous range of layers.
  /// Activations are passed between consecutive stages point-to-point.
  /// EXAMPLE: --pipeline-parallel-size 2
  int pipeline_parallel_size = 1;

  /// TODO(gc):
  int context_parallel_size = 1;

  /// Tensor parallelism: ranks that each hold a slice of every layer's weights
  /// (attention heads, MLP columns/rows). Layers finish with an all-reduce
  /// across the group.
  /// EXAMPLE: --tensor-parallel-size 2
  int tensor_parallel_size = 1;

  int expert_parallel_size = 1;

  /// Expert parallelism: place whole MoE experts on different ranks instead of
  /// sharding each expert with TP. Tokens are routed with all-to-all
  /// dispatch/combine. This is not an independent axis: the EP group reuses the
  /// existing `dp * tp` ranks within each pipeline stage (see
  /// ExpertParallelSize()). Has no effect on dense models.
  /// EXAMPLE: --enable-expert-parallel
  bool enable_expert_parallel = false;

  /// @}

  /// @name Rank coordinates
  /// This rank's coordinate within each group, in [0, corresponding size).
  /// Supplied by the topology/session layer for rank-local callers; the
  /// single-process default (all zero) needs nothing.
  /// @{
  int tensor_parallel_rank = 0;
  int pipeline_parallel_rank = 0;
  int data_parallel_rank = 0;
  /// @}

  /// true : attention is data-parallel across `dp` ranks (each has its own batch and KV),
  ///        FFN/MoE spans the dp*tp plane. Replicas are therefore lockstep-coupled.
  /// false: DP replicas are fully independent (no cross-DP communication).
  bool dp_attention = false;

  /// Megatron-style sequence parallelism inside the TP region (needs tp > 1).
  bool sequence_parallel = false;

  /// Mixed-radix order of mesh axes, outermost -> innermost. Innermost varies fastest in rank
  /// numbering, so by default TP groups are contiguous ranks (intra-node, NVLink-friendly).
  std::array<Axis, 4> axis_order{Axis::kDataParallel, Axis::kPipelineParallel,
                                 Axis::kContextParallel, Axis::kTensorParallel};

  /// Layers per pipeline stage. Empty = balanced split. Size must equal pp when non-empty.
  std::vector<int> pp_layer_split;

  /// Otherwise Topology::build rejects TP groups spanning nodes
  bool allow_cross_node_tp = false;

  /// \brief Total ranks (one CUDA device each in multi-GPU serving).
  [[nodiscard]] constexpr int WorldSize() const noexcept {
    return tensor_parallel_size * pipeline_parallel_size * context_parallel_size *
           data_parallel_size;
  }

  /// \brief Number of ranks that share MoE FFN work for one (pp, cp) coordinate.
  /// \return
  [[nodiscard]] constexpr int PlaneSize() const noexcept {
    return dp_attention ? tensor_parallel_size * data_parallel_size : tensor_parallel_size;
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

    if (tensor_parallel_rank < 0 || pipeline_parallel_rank < 0 || data_parallel_rank < 0) {
      return InvalidArgumentError("rank out of range for its group size");
    }

    return OkStatus();
  }
};

struct ClusterSpec {
  int num_nodes = 1;
  int gpus_per_node = 1;
  std::string rendezvous_endpoint;  // host:port of the bootstrap store
  std::chrono::seconds init_timeout{300};
};

}  // namespace inferx

#endif  // INFERX_CONFIG_PARALLEL_CONFIG_H_
