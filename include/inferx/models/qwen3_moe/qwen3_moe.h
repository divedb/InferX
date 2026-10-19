#ifndef INFERX_MODELS_QWEN3_MOE_QWEN3_MOE_H_
#define INFERX_MODELS_QWEN3_MOE_QWEN3_MOE_H_

#include <cstdint>

#include "inferx/config/parallel_config.h"
#include "inferx/core/device.h"
#include "inferx/core/dtype.h"
#include "inferx/core/status.h"
#include "inferx/core/tensor.h"
#include "inferx/models/layers/fused_moe.h"
#include "inferx/models/layers/linear.h"
#include "inferx/models/layers/mlp.h"
#include "inferx/dist/shard.h"
#include "inferx/models/model.h"
#include "inferx/models/qwen3_moe/qwen3_moe_config.h"
#include "inferx/models/weight_loader.h"

namespace inferx {
namespace models {
namespace qwen3_moe {

/// \brief Qwen3-MoE feed-forward block: a replicated router, the expert
///        bank, and (when the checkpoint has one) a sigmoid-gated shared
///        expert whose output adds onto the routed mixture.
///
/// Routing policy belongs to this class; `FusedExperts` executes
/// whatever (indices, weights) it is handed and knows nothing about
/// softmaxes or top-k. The shared expert path mirrors vLLM's: shared MLP
/// output multiplied by sigmoid(shared_expert_gate(x)), summed with the
/// routed output.
class Qwen3Moe {
 public:
  Qwen3Moe() = default;

  static StatusOr<Qwen3Moe> Create(const Qwen3MoeConfig& config,
                                   const ParallelConfig& parallel, DataType dtype,
                                   Device device);

  /// \brief Routes `hidden` and runs experts plus shared expert.
  ///
  /// \param out     [tokens, hidden] combined output; must not alias.
  Status Forward(ForwardContext& ctx, const Tensor& hidden, Tensor& out) const;

  /// \brief Binds `prefix.gate.weight`, `prefix.experts.M.*`, and the
  ///        shared-expert names when present.
  Status BindWeights(const std::string& prefix, WeightMap& weights);

  /// \brief Declarative manifest: the replicated router, one expert-kind
  ///        entry per expert (gate/up rows and down columns split over the
  ///        MoE-TP group, experts partitioned over Ep), and the shared
  ///        expert's dense-MLP names when present.
  [[nodiscard]] std::vector<ParamDecl> DescribeParams(const std::string& prefix) const;

 private:
  Qwen3Moe(Linear router, FusedExperts experts, std::optional<MLP> shared,
           std::optional<Linear> shared_gate, ops::RoutingConfig routing, int64_t hidden,
           int64_t tp_rank, int64_t num_experts, int64_t global_intermediate,
           int64_t shared_global_intermediate)
      : router_(std::move(router)),
        experts_(std::move(experts)),
        shared_expert_(std::move(shared)),
        shared_expert_gate_(std::move(shared_gate)),
        routing_(routing),
        hidden_(hidden),
        tp_rank_(tp_rank),
        num_experts_(num_experts),
        global_intermediate_(global_intermediate),
        shared_global_intermediate_(shared_global_intermediate) {}

  Linear router_;  ///< [experts, hidden], replicated across ranks.
  FusedExperts experts_;
  /// Present iff the checkpoint declares a shared expert.
  std::optional<MLP> shared_expert_;
  /// Present iff the shared expert is sigmoid-gated.
  std::optional<Linear> shared_expert_gate_;
  ops::RoutingConfig routing_;
  int64_t hidden_ = 0;
  int64_t tp_rank_ = 0;
  // Checkpoint extents for DescribeParams.
  int64_t num_experts_ = 0;
  int64_t global_intermediate_ = 0;
  int64_t shared_global_intermediate_ = 0;
};

}  // namespace qwen3_moe
}  // namespace models
}  // namespace inferx

#endif  // INFERX_MODELS_QWEN3_MOE_QWEN3_MOE_H_
