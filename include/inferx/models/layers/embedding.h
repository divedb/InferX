#ifndef INFERX_MODELS_LAYERS_EMBEDDING_H_
#define INFERX_MODELS_LAYERS_EMBEDDING_H_

#include <cstdint>

#include "inferx/core/status.h"
#include "inferx/core/tensor.h"
#include "inferx/dist/comm.h"
#include "inferx/ops/op_context.h"

namespace inferx::models::layers {

/// \brief Token embedding lookup, optionally vocabulary-sharded and scaled.
///
/// Owns one rank's row shard of the embedding table ([vocab/size, hidden]).
/// Under tensor parallelism, ids outside this rank's shard gather as zeros
/// and the all-reduce folds the ranks' masked partials into the full lookup
/// (VocabParallelEmbedding). The activation scale multiplies after the
/// gather (Gemma: sqrt(hidden)).
class TokenEmbedding {
 public:
  /// \param weight  [vocab_shard, hidden] rows of this rank.
  /// \param vocab_total  Full vocabulary width across all ranks.
  /// \param scale   Activation multiplier applied after the gather.
  TokenEmbedding(Tensor weight, int64_t vocab_total, float scale = 1.0f);

  /// \brief Gathers `token_ids` into `out` ([num_tokens, hidden]).
  ///
  /// \param token_ids [num_tokens] int32 ids on the context's device.
  /// \param out       Preallocated output buffer; the caller owns it so
  ///                  captured graphs see stable addresses.
  Status Embed(const Tensor& token_ids, Tensor& out, ops::OpContext& ctx,
               dist::CommBackend& comm) const;

  const Tensor& weight() const { return weight_; }
  int64_t vocab_total() const { return vocab_total_; }

 private:
  Tensor weight_;
  int64_t vocab_total_;
  float scale_;
};

}  // namespace inferx::models::layers

#endif  // INFERX_MODELS_LAYERS_EMBEDDING_H_
