/// \file
/// \brief Grouped-query causal attention: shared QKV projection, Q/K
/// normalization, RoPE, paged attention, and the output projection. The
/// layer receives small, concrete options and never inspects architecture
/// names or checkpoint JSON.

#ifndef INFERX_MODELS_LAYERS_ATTENTION_H_
#define INFERX_MODELS_LAYERS_ATTENTION_H_

#include <cstdint>
#include <optional>
#include <utility>

#include "inferx/config/parallel_config.h"
#include "inferx/core/status.h"
#include "inferx/core/tensor.h"
#include "inferx/models/layers/linear.h"
#include "inferx/models/model.h"
#include "inferx/models/requirements.h"
#include "inferx/ops/flash_attention.h"
#include "inferx/ops/op_context.h"
#include "inferx/ops/rotary.h"

namespace inferx::models::layers {

/// \brief Extra learned output projection attached to attention.
enum class OutputGate {
  kNone,     ///< Plain attention output.
  kSigmoid,  ///< Sigmoid-gated output; doubles q_proj rows (Qwen3-Next).
};

/// \brief Options of one grouped-query attention layer.
struct AttentionConfig {
  int64_t query_heads = 0;  ///< Total query heads, before any sharding.
  int64_t kv_heads = 0;     ///< Key/value heads; divides `query_heads`.
  int64_t head_dim = 0;     ///< Per-head dimension.
  bool qk_norm = false;     ///< Per-head RMSNorm on q and k.
  /// Q/K head norms scale by (1 + w) (Gemma, Qwen3-Next).
  bool qk_norm_plus_one = false;
  bool qkv_bias = false;    ///< Biases on the q/k/v projections.
  bool output_bias = false; ///< Bias on the output projection.
  /// Attention logit scale; 0 selects 1/sqrt(head_dim). Gemma derives it
  /// from query_pre_attn_scalar.
  float scale_override = 0.0f;
  ops::RotaryParams rotary;  ///< Finalized rotary parameters.
  int64_t sliding_window = 0;  ///< Tokens; 0 disables windowing.
  OutputGate output_gate = OutputGate::kNone;
  bool sinks = false;  ///< Per-head attention sink added to the denominator.
};

/// \brief Attention projections; shapes follow AttentionConfig.
///
/// `packed_qkv` is the fused projection of the shared QKV loader:
/// rank-local rows in block-contiguous order [query | key | value], where
/// the query section doubles to [query | gate] rows for a gated output.
/// Per-projection weights are views into that allocation sharing its
/// storage. `qkv_bias` matches the packed row layout; `sinks` carries
/// per-head bf16 sink logits when the architecture uses them.
struct AttentionWeights {
  Tensor packed_qkv;  ///< [query_rows + 2*kv_rows, hidden] fused rows.
  LinearWeights query;    ///< View of packed_qkv's query rows.
  LinearWeights key;      ///< View of packed_qkv's key rows.
  LinearWeights value;    ///< View of packed_qkv's value rows.
  LinearWeights output;   ///< [hidden, query_heads * head_dim]
  std::optional<Tensor> query_norm;  ///< [head_dim]; present only with qk_norm.
  std::optional<Tensor> key_norm;    ///< [head_dim]; present only with qk_norm.
  std::optional<Tensor> qkv_bias;    ///< Packed rows [q (| gate) | k | v].
  std::optional<Tensor> output_bias; ///< [hidden]; present only when biased.
  std::optional<Tensor> sinks;       ///< [query_heads] bf16; gpt-oss.
};

/// \brief Reusable attention workspace, sized once by the decoder stack.
///        Kernel-selection state lives in the ops-owned `plan`; the stack
///        only sizes buffers.
struct AttentionWorkspace {
  std::optional<Tensor> query;     ///< [max_tokens * max_query_width] flat.
  std::optional<Tensor> gate;      ///< Present only for gated outputs.
  std::optional<Tensor> key;       ///< [max_tokens * max_kv_width] flat.
  std::optional<Tensor> value;     ///< [max_tokens * max_kv_width] flat.
  std::optional<Tensor> attn_out;  ///< [max_tokens * max_query_width] flat.
  ops::AttentionPlanWorkspace plan;  ///< Kernel planning and selection state.
};

/// \brief Projects, qk-normalizes, rotates, caches, attends, and projects
///        back into `mixed_out` ([rows, hidden]).
///
/// \param packed_buffer  Shared packed-projection workspace the stack
///        provides; the fused QKV GEMM lands there before SplitQkv.
/// \param norm_eps  The layer's normalization epsilon, reused for q/k head
///        norms.
Status RunAttention(const AttentionConfig& config, const AttentionWeights& weights,
                    const Tensor& normed, float norm_eps, const AttentionBatch& batch,
                    const PagedKvState& kv_state, const KvBlockPool& pool,
                    AttentionWorkspace& ws, Tensor* packed_buffer, ops::OpContext& ctx,
                    Tensor& mixed_out);

/// \brief One rank's slice of the fused QKV projection.
///
/// All fields are rank-local except the replication bookkeeping. Row counts
/// describe the packed weight this rank loads and executes:
/// [query_rows | kv_rows | kv_rows] x hidden, block-contiguous.
struct QkvParallelGeometry {
  int64_t query_heads = 0;      ///< Query heads on this rank.
  int64_t kv_heads = 0;         ///< Key/value heads on this rank; at least 1.
  int64_t head_dim = 0;
  int64_t query_rows = 0;  ///< query_heads * head_dim (* 2 when gated).
  int64_t kv_rows = 0;     ///< kv_heads * head_dim.
  int64_t kv_head_replicas = 1;  ///< Ranks sharing each KV head when total < size.
  int64_t kv_shard = 0;          ///< Which KV-head shard this rank loads.
};

/// \brief Derives one rank's geometry from the total attention
///        configuration. Query heads divide across ranks (total must be
///        divisible). KV heads divide while total >= tensor_parallel_size;
///        below it each rank takes one KV head and `kv_head_replicas` ranks
///        replicate it. A gated output doubles the query rows this rank
///        loads.
StatusOr<QkvParallelGeometry> ShardQkv(const AttentionConfig& total,
                                       const ParallelConfig& parallel);

}  // namespace inferx::models::layers

#endif  // INFERX_MODELS_LAYERS_ATTENTION_H_
