#ifndef INFERX_MODELS_LAYERS_LM_HEAD_H_
#define INFERX_MODELS_LAYERS_LM_HEAD_H_

#include <cstdint>
#include <optional>

#include "inferx/core/status.h"
#include "inferx/core/tensor.h"
#include "inferx/dist/comm.h"
#include "inferx/ops/op_context.h"

namespace inferx::models::layers {

/// \brief Vocabulary projection over selected hidden-state rows.
///
/// `weight` is a [rows of vocab shard, hidden] matrix that may alias the
/// token embedding (tied parameters are explicit aliases). Under tensor
/// parallelism each rank computes [rows, shard] logits over its slice of
/// the vocabulary; the gather lays all shards side by side into
/// `vocab_total` width before sampling.
class LanguageModelHead {
 public:
  explicit LanguageModelHead(Tensor w) : weight(std::move(w)) {}

  int capacity = 0;  ///< Stable row capacity for graph replay.
  Tensor weight;     ///< [rows of vocab shard, hidden]; may alias the embedding.
  /// Full vocabulary width. Equals the shard width on a single rank; under
  /// tensor parallelism the ranks' logits all-gather into this width
  /// before sampling.
  int64_t vocab_total = 0;

  /// \brief Row and logits workspace bytes held on device.
  int64_t workspace_bytes() const;

  /// \brief Preallocates the row and logits workspace for `rows` rows.
  ///        Explicit counterpart to the first-forward lazy allocation;
  ///        Forward still grows the workspace if a larger batch appears.
  Status Prepare(int64_t rows, DeviceId device);

  /// \brief Gathers `rows`, then projects them to [rows, vocab] logits.
  /// The row buffer and logits borrow workspace allocated on first use;
  /// capacity is reserved up front so later batch growth never invalidates
  /// the row buffer referenced by an earlier decode graph.
  StatusOr<Tensor> Forward(const Tensor& hidden, const Tensor& rows, ops::OpContext& ctx,
                           dist::CommBackend& comm);

 private:
  bool workspace_ready_ = false;
  std::optional<Tensor> rows_, logits_, full_logits_;
};

}  // namespace inferx::models::layers

#endif  // INFERX_MODELS_LAYERS_LM_HEAD_H_
