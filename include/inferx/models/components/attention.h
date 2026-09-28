/// \file
/// \brief Attention token mixer: configuration, weights, and execution.

#ifndef INFERX_MODELS_COMPONENTS_ATTENTION_H_
#define INFERX_MODELS_COMPONENTS_ATTENTION_H_

#include <cstdint>
#include <optional>
#include <utility>

#include "inferx/core/tensor.h"
#include "inferx/models/components/linear.h"
#include "inferx/models/components/rope.h"
#include "inferx/models/state.h"
#include "inferx/ops/execution_context.h"
#include "inferx/ops/flash_attention.h"

namespace inferx {

struct AttentionBatch;  // defined in inferx/models/model.h
class DiagnosticTrace;  // src-private diagnostic helper

namespace components {

/// \brief Extra learned output projection attached to attention.
enum class OutputGate {
  kNone,     ///< Plain attention output.
  kSigmoid,  ///< Sigmoid-gated output; doubles q_proj rows (Qwen3-Next).
};

/// \brief Grouped-query causal attention.
struct AttentionConfig {
  int64_t query_heads = 0;  ///< Total query heads, before any sharding.
  int64_t kv_heads = 0;     ///< Key/value heads; divides query_heads.
  int64_t head_dim = 0;
  bool qk_norm = false;      ///< Per-head RMSNorm on q and k.
  bool qkv_bias = false;     ///< Biases on the q/k/v projections.
  bool output_bias = false;  ///< Bias on the output projection.
  RotaryConfig rotary;
  int64_t sliding_window = 0;  ///< Tokens; 0 disables windowing.
  OutputGate output_gate = OutputGate::kNone;
};

/// \brief Gated DeltaNet linear attention (Qwen3-Next recurrent layers).
struct GatedDeltaNetConfig {
  int64_t key_heads = 0;
  int64_t value_heads = 0;
  int64_t key_dim = 0;
  int64_t value_dim = 0;
  int64_t conv_kernel_size = 0;
};

/// \brief Attention projections; shapes follow AttentionConfig.
///
/// `packed_qkv` is the fused projection of the QKVParallelLinear component:
/// rank-local rows in block-contiguous order [query | key | value]. The
/// per-projection weights are views into that allocation and share its
/// storage.
struct AttentionWeights {
  Tensor packed_qkv;                 ///< [query_rows + 2*kv_rows, hidden] fused rows.
  LinearWeights query;               ///< View of packed_qkv's query rows.
  LinearWeights key;                 ///< View of packed_qkv's key rows.
  LinearWeights value;               ///< View of packed_qkv's value rows.
  LinearWeights output;              ///< [hidden, query_heads * head_dim]
  std::optional<Tensor> query_norm;  ///< [head_dim]; present only with qk_norm.
  std::optional<Tensor> key_norm;    ///< [head_dim]; present only with qk_norm.
};

/// \brief Reusable attention workspace, sized once by the decoder stack.
///
/// Kernel selection state lives in the ops-owned `plan`; the stack only
/// sizes the buffers.
struct AttentionWorkspace {
  std::optional<Tensor> query;       ///< [max_tokens * max_query_width] flat.
  std::optional<Tensor> key;         ///< [max_tokens * max_kv_width] flat.
  std::optional<Tensor> value;       ///< [max_tokens * max_kv_width] flat.
  std::optional<Tensor> attn_out;    ///< [max_tokens * max_query_width] flat.
  ops::AttentionPlanWorkspace plan;  ///< Kernel planning and selection state.
};

/// \brief Projects, qk-normalizes, rotates, caches, attends, and projects
///        back into `mixed_out` ([rows, hidden]).
///
/// Concrete and non-virtual: called once per layer per step from the decoder
/// stack. `packed_buffer` is the shared packed-projection workspace the stack
/// always provides; the fused QKV GEMM lands there before SplitQkv. `norm_eps`
/// is the layer's normalization epsilon, reused for q/k head norms.
Status RunAttention(const AttentionConfig& config, const AttentionWeights& weights,
                    const Tensor& normed, float norm_eps, const AttentionBatch& batch,
                    const PagedKvState& kv_state, const KvBlockPool& pool,
                    AttentionWorkspace& ws, Tensor* packed_buffer, ops::ExecutionContext& ctx,
                    DiagnosticTrace* trace, std::string_view prefix, Tensor& mixed_out);

enum class QkvBias { kDisabled, kEnabled };
enum class QkNorm { kNone, kRmsNorm };

/// A concrete attention component selected by model traits. Kernel dispatch
/// and workspace planning stay in the shared operations.
template <QkvBias Bias, QkNorm Norm, RopeStyle Rope>
class GqaAttention {
 public:
  using Config = AttentionConfig;
  using Weights = AttentionWeights;
  static constexpr bool kQkvBias = Bias == QkvBias::kEnabled;
  static constexpr bool kQkNorm = Norm == QkNorm::kRmsNorm;
  static constexpr RopeStyle kRope = Rope;

  GqaAttention(Config config, Weights weights)
      : config_(std::move(config)), weights_(std::move(weights)) {}

  Status Forward(const Tensor& input, float norm_eps, const AttentionBatch& batch,
                 const PagedKvState& state, const KvBlockPool& pool,
                 AttentionWorkspace& workspace, Tensor* packed, ops::ExecutionContext& ctx,
                 DiagnosticTrace* trace, std::string_view prefix, Tensor& output) const {
    return RunAttention(config_, weights_, input, norm_eps, batch, state, pool, workspace,
                        packed, ctx, trace, prefix, output);
  }

 private:
  Config config_;
  Weights weights_;
};

}  // namespace components
}  // namespace inferx

#endif  // INFERX_MODELS_COMPONENTS_ATTENTION_H_
