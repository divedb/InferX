#include "inferx/models/lm/causal_lm.h"

#include <utility>

#include "inferx/ops/gather.h"
#include "inferx/ops/linear.h"

namespace inferx::lm {

StatusOr<Tensor> LanguageModelHead::Forward(const Tensor& hidden, const Tensor& rows,
                                            ops::ExecutionContext& ctx) {
  if (!weight.IsDefined() || hidden.Rank() != 2 || rows.Rank() != 1 ||
      weight.Rank() != 2 || hidden.Dim(1) != weight.Dim(1) ||
      rows.GetDataType() != DataType::kInt32 || hidden.Device() != ctx.device() ||
      rows.Device() != ctx.device() || weight.Device() != ctx.device()) {
    return InvalidArgumentError("invalid language-model head inputs");
  }
  const int64_t count = rows.Numel();
  if (count == 0) return InvalidArgumentError("language-model head needs at least one row");
  if (!workspace_ready_ || rows_.Dim(0) < count) {
    // Reserve all sequence slots before any capture: later batch growth must
    // not invalidate the row buffer referenced by an earlier decode graph.
    const int64_t reserve = std::max<int64_t>(count, capacity);
    INFERX_ASSIGN_OR_RETURN(
        rows_, Tensor::Empty(DataType::kBFloat16, Shape({reserve, weight.Dim(1)}), ctx.device()));
    INFERX_ASSIGN_OR_RETURN(
        logits_, Tensor::Empty(DataType::kBFloat16, Shape({reserve, weight.Dim(0)}), ctx.device()));
    workspace_ready_ = true;
  }
  INFERX_ASSIGN_OR_RETURN(Tensor row_batch, rows_.Slice(0, count));
  INFERX_RETURN_IF_ERROR(ops::GatherRows(ctx, hidden, rows, row_batch));
  INFERX_ASSIGN_OR_RETURN(Tensor logits, logits_.Slice(0, count));
  INFERX_RETURN_IF_ERROR(ops::Linear(ctx, row_batch, weight, logits));
  return logits;
}

CausalLM::CausalLM(std::unique_ptr<Decoder> decoder, LanguageModelHead head, int max_seqs)
    : decoder_(std::move(decoder)), head_(std::move(head)), max_seqs_(max_seqs) { head_.capacity = max_seqs; }

StatusOr<Tensor> CausalLM::Forward(const ModelInput& input, ModelState& state,
                                   ops::ExecutionContext& ctx) {
  if (input.attention.num_seqs <= 0 || input.attention.num_seqs > max_seqs_ ||
      input.logit_rows.Rank() != 1 || input.logit_rows.Numel() != input.attention.num_seqs ||
      input.logit_rows.GetDataType() != DataType::kInt32 ||
      input.logit_rows.Device() != ctx.device()) {
    return InvalidArgumentError("invalid requested language-model output rows");
  }
  DecoderInput decoder_input{input.token_ids, {}, input.attention};
  INFERX_ASSIGN_OR_RETURN(Tensor hidden, decoder_->Forward(decoder_input, state, ctx));
  return head_.Forward(hidden, input.logit_rows, ctx);
}

}  // namespace inferx::lm
