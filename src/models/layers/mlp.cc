#include "inferx/models/layers/mlp.h"

#include <utility>

#include "inferx/ops/elementwise.h"
#include "inferx/ops/linear.h"

namespace inferx {
namespace models {

StatusOr<MLP> MLP::Create(int64_t hidden, int64_t full_intermediate, int64_t tp_size,
                          int64_t tp_rank, ops::Activation act, DataType dtype,
                          Device device) {
  if (hidden <= 0 || full_intermediate <= 0 || tp_size <= 0 || tp_rank < 0 ||
      tp_rank >= tp_size || full_intermediate % tp_size != 0) {
    return InvalidArgumentError("MLP geometry is inconsistent: hidden ", hidden,
                                ", intermediate ", full_intermediate, ", tp ", tp_size,
                                " rank ", tp_rank);
  }
  const int64_t intermediate = full_intermediate / tp_size;
  INFERX_ASSIGN_OR_RETURN(Linear gate_up,
                          Linear::Create(hidden, 2 * intermediate, dtype, device));
  INFERX_ASSIGN_OR_RETURN(Linear down, Linear::Create(intermediate, hidden, dtype, device));
  return MLP(std::move(gate_up), std::move(down), intermediate, act);
}

Status MLP::Forward(ops::OpContext& ctx, Allocator& scratch, CommBackend& comm,
                    const Tensor& x, Tensor& out) const {
  const int64_t tokens = x.Dim(0);
  INFERX_ASSIGN_OR_RETURN(Tensor fused,
                          Tensor::Empty(x.GetDataType(), Shape({tokens, 2 * intermediate_}),
                                        &scratch));
  INFERX_ASSIGN_OR_RETURN(Tensor gated,
                          Tensor::Empty(x.GetDataType(), Shape({tokens, intermediate_}),
                                        &scratch));

  INFERX_RETURN_IF_ERROR(gate_up_.Forward(ctx, x, fused));
  INFERX_RETURN_IF_ERROR(
      ops::PackedGatedActivation(ctx, fused, gated, act_));
  INFERX_RETURN_IF_ERROR(down_.Forward(ctx, gated, out));
  return comm.AllReduceSum(ctx, out);
}

}  // namespace models
}  // namespace inferx
