#include "inferx/models/layers/embedding.h"

#include <utility>

#include "inferx/ops/gather.h"

namespace inferx {
namespace models {

StatusOr<VocabParallelEmbedding> VocabParallelEmbedding::Create(int64_t full_vocab,
                                                               int64_t hidden_size,
                                                               int64_t tp_size, int64_t tp_rank,
                                                               DataType dtype,
                                                               Device device) {
  if (full_vocab <= 0 || hidden_size <= 0 || tp_size <= 0 || tp_rank < 0 ||
      tp_rank >= tp_size || full_vocab % tp_size != 0) {
    return InvalidArgumentError("embedding geometry is inconsistent: vocab ", full_vocab,
                                ", hidden ", hidden_size, ", tp ", tp_size, " rank ", tp_rank);
  }
  const int64_t shard = full_vocab / tp_size;
  INFERX_ASSIGN_OR_RETURN(
      Tensor weight, Tensor::Empty(dtype, Shape({shard, hidden_size}), device));
  return VocabParallelEmbedding(std::move(weight), full_vocab, tp_rank * shard);
}

Status VocabParallelEmbedding::Forward(ops::OpContext& ctx, CommBackend& comm,
                                       const Tensor& ids, Tensor& out) const {
  INFERX_RETURN_IF_ERROR(ops::GatherRowsRange(ctx, weight_, ids, out, shard_row_begin_));
  return comm.AllReduceSum(ctx, out);
}

}  // namespace models
}  // namespace inferx
