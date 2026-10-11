#ifndef INFERX_LAYERS_ATTENTION_H_
#define INFERX_LAYERS_ATTENTION_H_

#include <cstdint>
#include <optional>

#include "absl/types/span.h"
#include "inferx/cache/kv_block_pool.h"
#include "inferx/core/device.h"
#include "inferx/core/status.h"
#include "inferx/core/tensor.h"
#include "inferx/ops/attention.h"
#include "inferx/ops/flash_attention.h"
#include "inferx/ops/op_context.h"

namespace inferx {
namespace models {

/// \brief The batch plan one attention step executes against, shared by
///        every layer of the model.
///
/// This is the runtime's description of a scheduled step in the exact shape
/// the paged kernels consume: a flat token stream grouped by sequence via
/// CSR offsets, positions in each sequence's own coordinate space, and the
/// flattened page tables of every sequence's current KV extent. The runner
/// builds it once per step; layers only read it. Members are optional only
/// so the struct can be default-constructed and filled in place; every
/// field is present before a layer runs.
struct AttentionMetadata {
  /// [num_tokens] int32; sequence slot each token belongs to.
  std::optional<Tensor> batch_indices;
  /// [num_seqs + 1] int32; query extents (this step's new tokens).
  std::optional<Tensor> qo_indptr;
  /// [num_seqs + 1] int32; page-table extents of the full KV extent,
  /// including the tokens written this step.
  std::optional<Tensor> kv_indptr;
  /// [kv_pages] int32; block ids, sequence after sequence.
  std::optional<Tensor> kv_indices;
  /// [num_seqs] int32; tokens in each sequence's last page
  /// (((len - 1) % block_size) + 1).
  std::optional<Tensor> last_page_len;

  /// Host mirror of qo_indptr for tile planning; owned by the runner.
  absl::Span<const int32_t> host_qo_indptr;

  int num_tokens = 0;
  int num_seqs = 0;
};

/// \brief Core attention over the paged KV cache: geometry plus the
///        cache-write/attend sequence.
///
/// This module is everything every decoder-family attention shares and
/// nothing more: append k/v to the cache, then evaluate causal paged
/// attention for the step's queries. Projections, per-head norms, rope,
/// sliding-window config, and the output projection stay in the family's
/// attention class, which calls `Forward` with ready q/k/v. Kernel choice
/// (FlashInfer prefill tiles, split decode, generic fallback) is decided
/// inside the op layer from the metadata and workspace, never here.
class Attention {
 public:
  /// \brief Per-rank head geometry; `scale` is the qk product scale.
  struct Geometry {
    int64_t num_heads = 0;      ///< Query heads on this rank.
    int64_t num_kv_heads = 0;   ///< KV heads on this rank (GQA group divides).
    int64_t head_dim = 0;
    float scale = 1.0f;         ///< Typically 1 / sqrt(head_dim).
    int64_t sliding_window = 0; ///< 0 disables windowing.
  };

  Attention() = default;

  /// \brief Validates the geometry against kernel dispatcher support.
  static StatusOr<Attention> Create(Geometry geometry);

  /// \brief Appends k/v to layer `layer`'s cache, then attends.
  ///
  /// \param ctx        Execution context.
  /// \param attn       The step's batch plan.
  /// \param kv         The pool holding every layer's cache.
  /// \param layer      Decoder layer index whose cache views to use.
  /// \param plan       Runner-owned planning workspace, shared across layers.
  /// \param positions  [tokens] int32 token positions; selects each
  ///                    token's cache slot.
  /// \param q          [tokens, num_heads * head_dim] queries.
  /// \param k          [tokens, num_kv_heads * head_dim] keys, pre-roped.
  /// \param v          [tokens, num_kv_heads * head_dim] values.
  /// \param out        [tokens, num_heads * head_dim] attention output.
  /// \return           OK, or the first kernel's error.
  Status Forward(ops::OpContext& ctx, const AttentionMetadata& attn,
                 const KvBlockPool& kv, int64_t layer,
                 ops::AttentionPlanWorkspace& plan, const Tensor& positions,
                 const Tensor& q, const Tensor& k, const Tensor& v, Tensor& out) const;

  const Geometry& GetGeometry() const { return geometry_; }
  int64_t GroupSize() const { return geometry_.num_heads / geometry_.num_kv_heads; }

 private:
  explicit Attention(Geometry geometry) : geometry_(geometry) {}

  Geometry geometry_;
};

}  // namespace models
}  // namespace inferx

#endif  // INFERX_LAYERS_ATTENTION_H_
