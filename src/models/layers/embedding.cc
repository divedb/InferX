#include "inferx/models/layers/embedding.h"

#include "inferx/ops/elementwise.h"
#include "inferx/ops/gather.h"

namespace inferx::models::layers {

TokenEmbedding::TokenEmbedding(Tensor weight, int64_t vocab_total, float scale)
    : weight_(std::move(weight)), vocab_total_(vocab_total), scale_(scale) {}

Status TokenEmbedding::Embed(const Tensor& token_ids, Tensor& out, ops::OpContext& ctx,
                             dist::CommBackend& comm) const {
  if (token_ids.Rank() != 1 || token_ids.GetDataType() != DataType::kInt32 ||
      token_ids.Device() != ctx.Device() || out.Rank() != 2 ||
      out.Dim(0) != token_ids.Numel() || out.Dim(1) != weight_.Dim(1) ||
      out.Device() != ctx.Device() || weight_.Device() != ctx.Device()) {
    return InvalidArgumentError("invalid token embedding inputs");
  }
  if (vocab_total_ != comm.size() * weight_.Dim(0)) {
    return InvalidArgumentError("embedding shard and communicator topology disagree");
  }
  const bool sharded = comm.size() > 1;
  if (sharded) {
    // Vocab-parallel lookup: ids outside this rank's shard gather as zeros;
    // the all-reduce folds the ranks' masked partials.
    const int64_t offset = comm.rank() * weight_.Dim(0);
    INFERX_RETURN_IF_ERROR(ops::GatherRowsRange(ctx, weight_, token_ids, out, offset));
  } else {
    INFERX_RETURN_IF_ERROR(ops::GatherRows(ctx, weight_, token_ids, out));
  }
  INFERX_RETURN_IF_ERROR(comm.AllReduceSum(ctx, out));
  if (scale_ != 1.0f) {
    INFERX_RETURN_IF_ERROR(ops::MulScalar(ctx, out, scale_));
  }
  return OkStatus();
}

}  // namespace inferx::models::layers
