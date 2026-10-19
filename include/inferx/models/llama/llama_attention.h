#ifndef INFERX_MODELS_LLAMA_LLAMA_ATTENTION_H_
#define INFERX_MODELS_LLAMA_LLAMA_ATTENTION_H_

#include <cstdint>

#include "inferx/config/parallel_config.h"
#include "inferx/core/device.h"
#include "inferx/core/dtype.h"
#include "inferx/core/status.h"
#include "inferx/core/tensor.h"
#include "inferx/dist/shard.h"
#include "inferx/ops/attention/spec.h"
#include "inferx/models/layers/linear.h"
#include "inferx/models/layers/rotary_embedding.h"
#include "inferx/models/llama/llama_config.h"
#include "inferx/models/model.h"
#include "inferx/models/weight_loader.h"

namespace inferx {
namespace models {
namespace llama {

/// \brief Llama self-attention: fused QKV projection, neox rope directly on
///        q/k (no per-head norms), GQA over the paged cache, row-parallel
///        output projection.
class LlamaAttention {
 public:
  LlamaAttention() = default;

  static StatusOr<LlamaAttention> Create(const LlamaConfig& config,
                                         const ParallelConfig& parallel, DataType dtype,
                                         Device device);

  Status Forward(ForwardContext& ctx, const Tensor& positions, const Tensor& hidden,
                Tensor& out) const;

  /// \brief Binds `prefix.{q,k,v,o}_proj.weight`.
  Status BindWeights(const std::string& prefix, WeightMap& weights);

  /// \brief This layer's attention spec; the runner binds an executor for
  ///        it at init (docs/attention.md §6.1).
  [[nodiscard]] attn::AttentionSpec Spec() const;

  /// \brief Declarative parameter manifest for this layer's projections
  ///        (Q/K/V of the fused buffer leave `dst` empty; placement is
  ///        model-side, as in BindWeights).
  [[nodiscard]] std::vector<ParamDecl> DescribeParams(const std::string& prefix) const;

 private:
  LlamaAttention(Linear qkv_proj, RotaryEmbedding rope, Linear o_proj, int64_t local_heads,
                 int64_t local_kv_heads, int64_t head_dim, int64_t sliding_window,
                 int64_t tp_rank, int64_t hidden_size, int64_t global_q_rows,
                 int64_t global_kv_rows, int kv_replicate)
      : qkv_proj_(std::move(qkv_proj)),
        rope_(std::move(rope)),
        o_proj_(std::move(o_proj)),
        local_heads_(local_heads),
        local_kv_heads_(local_kv_heads),
        head_dim_(head_dim),
        sliding_window_(sliding_window),
        tp_rank_(tp_rank),
        hidden_size_(hidden_size),
        global_q_rows_(global_q_rows),
        global_kv_rows_(global_kv_rows),
        kv_replicate_(kv_replicate) {}

  Linear qkv_proj_;  ///< [(qh + 2*kvh) * hd, hidden], rows [q | k | v]
  RotaryEmbedding rope_;
  Linear o_proj_;    ///< [hidden, qh * hd] (rank's column shard)

  int64_t local_heads_ = 0;
  int64_t local_kv_heads_ = 0;
  int64_t head_dim_ = 0;
  /// 0 disables windowing (also stored in the spec's mask).
  int64_t sliding_window_ = 0;
  int64_t tp_rank_ = 0;

  // Checkpoint extents for DescribeParams; config-derived, rank-free.
  int64_t hidden_size_ = 0;
  int64_t global_q_rows_ = 0;
  int64_t global_kv_rows_ = 0;
  /// Ranks sharing one KV piece: tp / min(tp, num_key_value_heads).
  int kv_replicate_ = 1;
};

}  // namespace llama
}  // namespace models
}  // namespace inferx

#endif  // INFERX_MODELS_LLAMA_LLAMA_ATTENTION_H_
