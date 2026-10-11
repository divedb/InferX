#ifndef INFERX_LAYERS_FUSED_MOE_H_
#define INFERX_LAYERS_FUSED_MOE_H_

#include <cstdint>

#include "inferx/core/allocator.h"
#include "inferx/core/device.h"
#include "inferx/core/dtype.h"
#include "inferx/core/status.h"
#include "inferx/core/tensor.h"
#include "inferx/dist/comm.h"
#include "inferx/ops/elementwise.h"
#include "inferx/ops/op_context.h"

namespace inferx {
namespace models {

/// \brief Expert bank for routed MoE layers: every expert's SwiGLU weights
///        plus the dispatch/gather/execute/scatter execution over them.
///
/// Routing is deliberately outside this module. The owning model computes
/// `topk_indices`/`topk_weights` with whatever policy its family defines
/// (`ops::RouteTokens` today) and hands the results here. Execution groups
/// routed slots by expert, runs each occupied expert as dense GEMMs over
/// its slice (fused gate/up, gated activation, down projection), and
/// scatters weighted expert outputs back onto their token rows. Under
/// tensor parallelism each expert is column/row sharded like a dense MLP
/// and the partial outputs are all-reduced once per layer.
///
/// Expert weights are stored flattened so weight loading stays plain 2-D
/// row copies: `gate_up` is [experts * 2I, hidden] with each expert's rows
/// [gate_shard; up_shard], and `down` is [experts * hidden, I].
class FusedExperts {
 public:
  FusedExperts() = default;

  /// \brief Allocates the expert bank for this rank.
  ///
  /// \param num_experts       Expert count E.
  /// \param hidden            Residual-stream width.
  /// \param full_intermediate Per-expert intermediate size; divided by
  ///                         `tp_size`.
  /// \param tp_size           Tensor-parallel world size.
  /// \param tp_rank           This rank's index.
  /// \param act               Gating nonlinearity (Silu for Qwen3-MoE).
  /// \param dtype             Activation dtype; bfloat16 today.
  /// \param device            Destination device.
  static StatusOr<FusedExperts> Create(int64_t num_experts, int64_t hidden,
                                       int64_t full_intermediate, int64_t tp_size,
                                       int64_t tp_rank, ops::Activation act,
                                       DataType dtype, Device device);

  /// \brief Runs every occupied expert over the routed token slots.
  ///
  /// \param ctx           Execution context.
  /// \param scratch       Allocator for dispatch and expert buffers.
  /// \param comm          Tensor-parallel group; reduces partial outputs.
  /// \param hidden_states [rows, hidden] layer input.
  /// \param topk_indices  [rows * topk] int32 expert ids (from routing).
  /// \param topk_weights  [rows * topk] float32 routing weights.
  /// \param out           [rows, hidden] weighted expert output; zeroed
  ///                      here, must not alias `hidden_states`.
  /// \return              OK, or the first failing kernel.
  Status Forward(ops::OpContext& ctx, Allocator& scratch, CommBackend& comm,
                 const Tensor& hidden_states, const Tensor& topk_indices,
                 const Tensor& topk_weights, Tensor& out) const;

  /// [experts * 2I, hidden]; each expert's rows are [gate; up] shards.
  const Tensor& GateUpWeight() const { return gate_up_; }
  Tensor* MutableGateUpWeight() { return &gate_up_; }

  /// [experts * hidden, I]; each expert's rows hold its down shard.
  const Tensor& DownWeight() const { return down_; }
  Tensor* MutableDownWeight() { return &down_; }

  int64_t NumExperts() const { return num_experts_; }
  int64_t IntermediateShard() const { return intermediate_; }

 private:
  FusedExperts(Tensor gate_up, Tensor down, int64_t num_experts, int64_t intermediate,
               ops::Activation act)
      : gate_up_(std::move(gate_up)),
        down_(std::move(down)),
        num_experts_(num_experts),
        intermediate_(intermediate),
        act_(act) {}

  Tensor gate_up_;  ///< [E * 2I, hidden]
  Tensor down_;     ///< [E * hidden, I]
  int64_t num_experts_ = 0;
  int64_t intermediate_ = 0;
  ops::Activation act_ = ops::Activation::kSilu;
};

}  // namespace models
}  // namespace inferx

#endif  // INFERX_LAYERS_FUSED_MOE_H_
