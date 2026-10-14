#include "inferx/models/layers/fused_moe.h"

#include <utility>
#include <vector>

#include "inferx/ops/linear.h"
#include "inferx/ops/moe.h"

namespace inferx {
namespace models {

StatusOr<FusedExperts> FusedExperts::Create(int64_t num_experts, int64_t hidden,
                                            int64_t full_intermediate, int64_t tp_size,
                                            int64_t tp_rank, ops::Activation act,
                                            DataType dtype, Device device) {
  if (num_experts <= 0 || hidden <= 0 || full_intermediate <= 0 || tp_size <= 0 ||
      tp_rank < 0 || tp_rank >= tp_size || full_intermediate % tp_size != 0) {
    return InvalidArgumentError("expert bank geometry is inconsistent: experts ",
                                num_experts, ", hidden ", hidden, ", intermediate ",
                                full_intermediate, ", tp ", tp_size, " rank ", tp_rank);
  }
  const int64_t intermediate = full_intermediate / tp_size;
  // Flattened expert stacks: [E * 2I, H] and [E * H, I], so every weight
  // load is one plain 2-D row copy and every expert slice is a row slice.
  INFERX_ASSIGN_OR_RETURN(
      Tensor gate_up,
      Tensor::Empty(dtype, Shape({num_experts * 2 * intermediate, hidden}), device));
  INFERX_ASSIGN_OR_RETURN(
      Tensor down, Tensor::Empty(dtype, Shape({num_experts * hidden, intermediate}), device));
  return FusedExperts(std::move(gate_up), std::move(down), num_experts, intermediate, act);
}

Status FusedExperts::Forward(ops::OpContext& ctx, Allocator& scratch,
                             CommBackend& comm, const Tensor& hidden_states,
                             const Tensor& topk_indices, const Tensor& topk_weights,
                             Tensor& out) const {
  const int64_t rows = hidden_states.Dim(0);
  const int64_t hidden = hidden_states.Dim(1);
  const int64_t total = topk_indices.Numel();
  if (topk_weights.Numel() != total || out.Dim(0) != rows || out.Dim(1) != hidden) {
    return InvalidArgumentError("expert dispatch disagrees with the routed batch");
  }

  // Scatter-add accumulates every (token, expert) pair onto its row, so the
  // output starts zeroed on the stream.
  INFERX_RETURN_IF_ERROR(
      ctx.Runtime().Memory().MemsetAsync(out.Data(), uint8_t{0}, out.NBytes(), ctx.GetStream()));
  if (total == 0) return OkStatus();

  const DataType act_dtype = hidden_states.GetDataType();
  INFERX_ASSIGN_OR_RETURN(Tensor counts,
                          Tensor::Empty(DataType::kInt32, Shape({num_experts_}), &scratch));
  INFERX_ASSIGN_OR_RETURN(
      Tensor offsets, Tensor::Empty(DataType::kInt32, Shape({num_experts_ + 1}), &scratch));
  INFERX_ASSIGN_OR_RETURN(Tensor cursor,
                          Tensor::Empty(DataType::kInt32, Shape({num_experts_}), &scratch));
  INFERX_ASSIGN_OR_RETURN(
      Tensor token_rows, Tensor::Empty(DataType::kInt32, Shape({total}), &scratch));
  INFERX_ASSIGN_OR_RETURN(
      Tensor weights_by_slot, Tensor::Empty(DataType::kFloat32, Shape({total}), &scratch));
  INFERX_ASSIGN_OR_RETURN(
      Tensor routed, Tensor::Empty(act_dtype, Shape({total, hidden}), &scratch));
  INFERX_ASSIGN_OR_RETURN(
      Tensor gate_up_out,
      Tensor::Empty(act_dtype, Shape({total, 2 * intermediate_}), &scratch));
  INFERX_ASSIGN_OR_RETURN(
      Tensor gated, Tensor::Empty(act_dtype, Shape({total, intermediate_}), &scratch));
  INFERX_ASSIGN_OR_RETURN(
      Tensor expert_rows, Tensor::Empty(act_dtype, Shape({total, hidden}), &scratch));

  // Histogram + host offsets (synchronizes), then slot scatter. The offsets
  // are the GEMM row counts, which is why they must reach the host.
  INFERX_ASSIGN_OR_RETURN(
      std::vector<int64_t> host_offsets,
      ops::BuildExpertDispatch(ctx, topk_indices, topk_weights, num_experts_, counts,
                               offsets, cursor, token_rows, weights_by_slot));
  INFERX_RETURN_IF_ERROR(ops::GatherRoutedTokens(ctx, hidden_states, token_rows, routed));

  for (int64_t expert = 0; expert < num_experts_; ++expert) {
    const int64_t begin = host_offsets[expert];
    const int64_t count = host_offsets[expert + 1] - begin;
    if (count == 0) continue;

    INFERX_ASSIGN_OR_RETURN(Tensor in_rows, routed.Slice(begin, begin + count));
    INFERX_ASSIGN_OR_RETURN(Tensor fused_rows,
                            gate_up_out.Slice(begin, begin + count));
    INFERX_ASSIGN_OR_RETURN(Tensor gated_rows, gated.Slice(begin, begin + count));
    INFERX_ASSIGN_OR_RETURN(Tensor down_rows, expert_rows.Slice(begin, begin + count));

    // This expert's weight slices: rows [expert * 2I, (expert + 1) * 2I) of
    // the fused gate/up stack and [expert * H, (expert + 1) * H) of down.
    INFERX_ASSIGN_OR_RETURN(Tensor gate_up_w,
                            gate_up_.Slice(expert * 2 * intermediate_,
                                           (expert + 1) * 2 * intermediate_));
    INFERX_ASSIGN_OR_RETURN(Tensor down_w,
                            down_.Slice(expert * hidden, (expert + 1) * hidden));

    INFERX_RETURN_IF_ERROR(ops::Linear(ctx, in_rows, gate_up_w, fused_rows));
    INFERX_RETURN_IF_ERROR(
        ops::PackedGatedActivation(ctx, fused_rows, gated_rows, act_));
    INFERX_RETURN_IF_ERROR(ops::Linear(ctx, gated_rows, down_w, down_rows));
    INFERX_RETURN_IF_ERROR(ops::ScatterRoutedOutputs(ctx, expert_rows, token_rows,
                                                     weights_by_slot, begin, count, out));
  }

  // Under TP every rank computed its intermediate shard of each expert, so
  // the scattered sums are partial until this reduce.
  return comm.AllReduceSum(ctx, out);
}

}  // namespace models
}  // namespace inferx
