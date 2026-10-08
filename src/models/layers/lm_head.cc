#include "inferx/models/layers/lm_head.h"

#include <algorithm>

#include "inferx/ops/gather.h"
#include "inferx/ops/linear.h"

namespace inferx::models::layers {

Status LanguageModelHead::Prepare(int64_t rows, DeviceId device) {
  if (rows <= 0) return InvalidArgumentError("language-model head needs a positive capacity");
  if (workspace_ready_) {
    if (rows > rows_->Dim(0)) {
      return InvalidArgumentError("execution limits exceed the prepared head capacity");
    }
    return OkStatus();
  }
  const int64_t reserve = std::max<int64_t>(rows, capacity);
  INFERX_ASSIGN_OR_RETURN(
      rows_, Tensor::Empty(DataType::kBFloat16, Shape({reserve, weight.Dim(1)}), device));
  INFERX_ASSIGN_OR_RETURN(
      logits_, Tensor::Empty(DataType::kBFloat16, Shape({reserve, weight.Dim(0)}), device));
  const int64_t vocab = vocab_total > 0 ? vocab_total : weight.Dim(0);
  if (vocab != weight.Dim(0)) {
    INFERX_ASSIGN_OR_RETURN(
        full_logits_, Tensor::Empty(DataType::kBFloat16, Shape({reserve, vocab}), device));
  }
  workspace_ready_ = true;
  return OkStatus();
}

int64_t LanguageModelHead::workspace_bytes() const {
  int64_t bytes = 0;
  for (const auto* t : {&rows_, &logits_, &full_logits_}) {
    if (t->has_value()) bytes += (*t)->NBytes();
  }
  return bytes;
}

StatusOr<Tensor> LanguageModelHead::Forward(const Tensor& hidden, const Tensor& rows,
                                            ops::OpContext& ctx, dist::CommBackend& comm) {
  if (hidden.Rank() != 2 || rows.Rank() != 1 || weight.Rank() != 2 ||
      hidden.Dim(1) != weight.Dim(1) || rows.GetDataType() != DataType::kInt32 ||
      hidden.Device() != ctx.Device() || rows.Device() != ctx.Device() ||
      weight.Device() != ctx.Device()) {
    return InvalidArgumentError("invalid language-model head inputs");
  }
  const int64_t count = rows.Numel();
  if (count == 0) return InvalidArgumentError("language-model head needs at least one row");
  const int64_t vocab = vocab_total > 0 ? vocab_total : weight.Dim(0);
  if (vocab != comm.size() * weight.Dim(0)) {
    return InvalidArgumentError("language-model head and communicator topology disagree");
  }
  if (!workspace_ready_ || rows_->Dim(0) < count) {
    // Reserve all sequence slots before any capture: later batch growth must
    // not invalidate the row buffer referenced by an earlier decode graph.
    const int64_t reserve = std::max<int64_t>(count, capacity);
    INFERX_ASSIGN_OR_RETURN(
        rows_,
        Tensor::Empty(DataType::kBFloat16, Shape({reserve, weight.Dim(1)}), ctx.Device()));
    INFERX_ASSIGN_OR_RETURN(
        logits_,
        Tensor::Empty(DataType::kBFloat16, Shape({reserve, weight.Dim(0)}), ctx.Device()));
    if (vocab != weight.Dim(0)) {
      INFERX_ASSIGN_OR_RETURN(
          full_logits_,
          Tensor::Empty(DataType::kBFloat16, Shape({reserve, vocab}), ctx.Device()));
    }
    workspace_ready_ = true;
  }
  INFERX_ASSIGN_OR_RETURN(Tensor row_batch, rows_->Slice(0, count));
  INFERX_RETURN_IF_ERROR(ops::GatherRows(ctx, hidden, rows, row_batch));
  INFERX_ASSIGN_OR_RETURN(Tensor logits, logits_->Slice(0, count));
  INFERX_RETURN_IF_ERROR(ops::Linear(ctx, row_batch, weight, logits));
  // Vocab-parallel head: this rank computed [rows, shard] logits over its
  // slice of the vocabulary; the gather lays all shards side by side.
  Tensor full = logits;
  if (full_logits_.has_value()) {
    INFERX_ASSIGN_OR_RETURN(full, full_logits_->Slice(0, count));
  }
  INFERX_RETURN_IF_ERROR(comm.AllGatherLastDim(ctx, logits, full));
  return full;
}

}  // namespace inferx::models::layers
