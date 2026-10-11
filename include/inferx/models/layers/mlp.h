#ifndef INFERX_LAYERS_MLP_H_
#define INFERX_LAYERS_MLP_H_

#include <cstdint>

#include "inferx/core/device.h"
#include "inferx/core/dtype.h"
#include "inferx/core/status.h"
#include "inferx/core/tensor.h"
#include "inferx/dist/comm.h"
#include "inferx/models/layers/linear.h"
#include "inferx/ops/elementwise.h"
#include "inferx/ops/op_context.h"

namespace inferx {
namespace models {

/// \brief SwiGLU feed-forward: fused gate/up projection, gated activation,
///        row-parallel down projection.
///
/// Weights are held fused exactly as the GEMMs want them:
/// `gate_up` [2 * intermediate_shard, hidden] laid out [gate; up], and
/// `down` [hidden, intermediate_shard] holding this rank's column block of
/// the full [hidden, intermediate]. The all-reduce after `down` folds the
/// rank-partial sums (identity at world size one).
class MLP {
 public:
  MLP() = default;

  /// \brief Allocates both projections for this rank.
  ///
  /// \param hidden             Residual-stream width.
  /// \param full_intermediate  Checkpoint intermediate size.
  /// \param tp_size            Tensor-parallel world size; divides
  ///                           `full_intermediate`.
  /// \param tp_rank            This rank's index.
  /// \param act                Gating nonlinearity (Silu for Qwen/Llama).
  /// \param dtype              Activation dtype.
  /// \param device             Destination device.
  static StatusOr<MLP> Create(int64_t hidden, int64_t full_intermediate, int64_t tp_size,
                              int64_t tp_rank, ops::Activation act, DataType dtype,
                              Device device);

  /// \brief out = down(act(gate_up(x))).
  ///
  /// \param scratch  Allocator for the intermediate [tokens, 2I] and gated
  ///                 [tokens, I] activations.
  /// \param comm     Tensor-parallel group; reduces `down`'s partial sums.
  /// \param x        [tokens, hidden].
  /// \param out      [tokens, hidden]; must not alias `x`.
  Status Forward(ops::OpContext& ctx, Allocator& scratch, CommBackend& comm,
                 const Tensor& x, Tensor& out) const;

  const Linear& GateUp() const { return gate_up_; }
  Linear* MutableGateUp() { return &gate_up_; }
  const Linear& Down() const { return down_; }
  Linear* MutableDown() { return &down_; }
  int64_t IntermediateShard() const { return intermediate_; }

 private:
  MLP(Linear gate_up, Linear down, int64_t intermediate, ops::Activation act)
      : gate_up_(std::move(gate_up)),
        down_(std::move(down)),
        intermediate_(intermediate),
        act_(act) {}

  Linear gate_up_;  ///< [2 * intermediate, hidden], rows [gate; up]
  Linear down_;     ///< [hidden, intermediate_shard]
  int64_t intermediate_ = 0;
  ops::Activation act_ = ops::Activation::kSilu;
};

}  // namespace models
}  // namespace inferx

#endif  // INFERX_LAYERS_MLP_H_
