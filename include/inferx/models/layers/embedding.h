#ifndef INFERX_LAYERS_EMBEDDING_H_
#define INFERX_LAYERS_EMBEDDING_H_

#include <cstdint>

#include "inferx/core/allocator.h"
#include "inferx/core/device.h"
#include "inferx/core/dtype.h"
#include "inferx/core/status.h"
#include "inferx/core/tensor.h"
#include "inferx/dist/comm.h"
#include "inferx/ops/op_context.h"

namespace inferx {
namespace models {

/// \brief Token embedding with the vocabulary split across tensor-parallel
///        ranks.
///
/// Each rank holds rows [rank * shard, (rank + 1) * shard) of the table.
/// The lookup gathers in-range ids and zeroes the rest; one all-reduce over
/// the rank group completes every row, so ids never leave their owning rank
/// unsafely. With world size one this is a plain embedding lookup.
class VocabParallelEmbedding {
 public:
  VocabParallelEmbedding() = default;

  /// \brief Allocates this rank's row shard [vocab_shard, hidden].
  ///
  /// \param full_vocab  Total vocabulary; must divide `tp_size`.
  /// \param hidden_size Embedding width.
  /// \param tp_size     Tensor-parallel world size.
  /// \param tp_rank     This rank's index in [0, tp_size).
  /// \param dtype       Weight dtype; bfloat16 today.
  /// \param device      Destination device.
  static StatusOr<VocabParallelEmbedding> Create(int64_t full_vocab, int64_t hidden_size,
                                                 int64_t tp_size, int64_t tp_rank,
                                                 DataType dtype, Device device);

  /// \brief Looks up `ids` and (when sharded) reduces across ranks.
  ///
  /// \param ctx  Execution context.
  /// \param comm The tensor-parallel group.
  /// \param ids  [tokens] int32 token ids.
  /// \param out  [tokens, hidden] embeddings; must not alias the table.
  Status Forward(ops::OpContext& ctx, CommBackend& comm, const Tensor& ids,
                 Tensor& out) const;

  const Tensor& Weight() const { return weight_; }
  Tensor* MutableWeight() { return &weight_; }
  int64_t FullVocab() const { return full_vocab_; }
  int64_t ShardRowBegin() const { return shard_row_begin_; }

 private:
  VocabParallelEmbedding(Tensor weight, int64_t full_vocab, int64_t shard_row_begin)
      : weight_(std::move(weight)),
        full_vocab_(full_vocab),
        shard_row_begin_(shard_row_begin) {}

  Tensor weight_;  ///< [vocab_shard, hidden]
  int64_t full_vocab_ = 0;
  int64_t shard_row_begin_ = 0;
};

}  // namespace models
}  // namespace inferx

#endif  // INFERX_LAYERS_EMBEDDING_H_
