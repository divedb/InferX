#ifndef INFERX_MODELS_QWEN3_QWEN3_ATTENTION_H_
#define INFERX_MODELS_QWEN3_QWEN3_ATTENTION_H_

#include "inferx/config/parallel_config.h"
#include "inferx/core/device.h"
#include "inferx/core/dtype.h"
#include "inferx/core/status.h"
#include "inferx/core/tensor.h"
#include "inferx/ops/attention/spec.h"
#include "inferx/models/layers/linear.h"
#include "inferx/models/layers/rms_norm.h"
#include "inferx/models/layers/rotary_embedding.h"
#include "inferx/models/model.h"
#include "inferx/models/qwen3/qwen3_config.h"
#include "inferx/models/weight_loader.h"

namespace inferx {
namespace models {
namespace qwen3 {

/// \brief Qwen3 self-attention: fused QKV projection, per-head q/k RMS
///        norms, neox rope, grouped-query attention over the paged cache,
///        and a row-parallel output projection.
///
/// Qwen3-MoE layers share this exact block, so they compose it directly
/// rather than deriving from it: the family difference is the feed-forward,
/// not the attention.
class Qwen3Attention {
 public:
  Qwen3Attention() = default;

  /// \brief Allocates weights for this rank's head shard.
  static StatusOr<Qwen3Attention> Create(const Qwen3Config& config,
                                         const ParallelConfig& parallel, DataType dtype,
                                         Device device);

  /// \brief Projects, normalizes, rotates, attends, and projects back.
  ///
  /// \param ctx     Step context; scratch supplies the intermediates.
  /// \param hidden  [tokens, hidden] layer input.
  /// \param out     [tokens, hidden] attention output; may not alias.
  Status Forward(ForwardContext& ctx, const Tensor& positions, const Tensor& hidden,
                Tensor& out) const;

  /// \brief Binds `prefix.{q,k,v,o}_proj.weight` and
  ///        `prefix.{q,k}_norm.weight` for every rank's shard.
  Status BindWeights(const std::string& prefix, WeightMap& weights);

  int64_t LocalQueryWidth() const { return local_heads_ * head_dim_; }

  /// \brief This layer's attention spec: what the family writes and how it
  ///        attends. The runner binds an executor for it at init.
  [[nodiscard]] attn::AttentionSpec Spec() const;

 private:
  Qwen3Attention(Linear qkv_proj, RMSNorm q_norm, RMSNorm k_norm, RotaryEmbedding rope,
                 Linear o_proj, int64_t local_heads, int64_t local_kv_heads,
                 int64_t head_dim, int64_t sliding_window, int64_t tp_rank)
      : qkv_proj_(std::move(qkv_proj)),
        q_norm_(std::move(q_norm)),
        k_norm_(std::move(k_norm)),
        rope_(std::move(rope)),
        o_proj_(std::move(o_proj)),
        local_heads_(local_heads),
        local_kv_heads_(local_kv_heads),
        head_dim_(head_dim),
        sliding_window_(sliding_window),
        tp_rank_(tp_rank) {}

  Linear qkv_proj_;  ///< [(qh + 2*kvh) * hd, hidden], rows [q | k | v]
  RMSNorm q_norm_;   ///< Per-head query norm over head_dim.
  RMSNorm k_norm_;   ///< Per-head key norm over head_dim.
  RotaryEmbedding rope_;
  Linear o_proj_;    ///< [hidden, qh * hd] (rank's column shard)

  int64_t local_heads_ = 0;
  int64_t local_kv_heads_ = 0;
  int64_t head_dim_ = 0;
  /// 0 disables windowing (also stored in the spec's mask).
  int64_t sliding_window_ = 0;
  int64_t tp_rank_ = 0;
};

}  // namespace qwen3
}  // namespace models
}  // namespace inferx

#endif  // INFERX_MODELS_QWEN3_QWEN3_ATTENTION_H_
