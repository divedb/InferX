#pragma once
#include <optional>

#include "absl/types/span.h"
#include "inferx/ops/attention.h"
namespace inferx::ops {
struct FlashDecodeWorkspace {
  // Eight partitions per request, up to 16 requests. Shared across layers.
  static constexpr int kPartitions = 8;
  static constexpr int kMaxBatch = 16;
  Tensor plan, values, scores;
};

/// \brief Kernel-selection and planning state for paged attention.
///
/// The engine-facing attention entry points own their kernel choice: which
/// implementation runs is decided from batch geometry and workspace
/// contents, never requested by the caller. The flash-attention plan is
/// rebuilt when the step's GQA group changes; the decode workspace, when
/// allocated, enables the split-decode kernel for small pure-decode steps.
struct AttentionPlanWorkspace {
  std::optional<Tensor> plan;  ///< int32 [3 * max_tokens + 1] plan, once sized.
  std::optional<FlashDecodeWorkspace> decode;  ///< Present iff split decode is enabled.
  /// Selection state for the current step; updated by BeginAttentionStep
  /// and PagedAttention, meaningless between steps.
  int planned_group = 0;
  int attention_tiles = 0;
  int prefill_tile_rows = 64;
  bool split_decode_selected = false;
};

/// \brief Marks a new attention step and selects this step's decode kernel.
///
/// Chooses the split-decode kernel when its workspace is allocated and the
/// batch is pure decode within its capacity, else the shared paged path.
Status BeginAttentionStep(ExecutionContext& ctx, const Tensor& kv_indptr,
                          const Tensor& last_page_len, int64_t block_size, int num_tokens,
                          int num_seqs, AttentionPlanWorkspace& ws);

/// \brief Runs causal paged attention for one layer, kernels chosen
///        internally.
///
/// Rebuilds the flash-attention plan when the GQA group changes, then
/// dispatches to the kernel BeginAttentionStep selected for the step.
/// `host_qo_indptr` mirrors `qo_indptr` for host-side tile planning.
Status PagedAttention(ExecutionContext& ctx, const Tensor& q, const Tensor& qo_indptr,
                      const Tensor& kv_indptr, const Tensor& kv_indices,
                      const Tensor& last_page_len, absl::Span<const int32_t> host_qo_indptr,
                      int num_seqs, const Tensor& key_cache, const Tensor& value_cache,
                      int64_t block_size, const AttentionParams& params,
                      AttentionPlanWorkspace& ws, Tensor& attn_out);

Status PrepareFlashDecode(ExecutionContext& ctx, const Tensor& kv_indptr,
                          const Tensor& last_page_len, int block_size,
                          FlashDecodeWorkspace& workspace);
// Graph-safe GPU planning, reused across layers. Inputs must describe positive
// query lengths and valid, nonempty KV page tables; each query chunk is the
// suffix of its sequence's current KV. Host/device metadata must agree.
// `tiles` must equal sum(ceil(query_length * group / tile_rows)). The caller
// owns validating device metadata values and keeping buffers alive on stream.
// Supported BF16 full causal geometry: head_dim 64/128/256, GQA ratio 1..32.
// Both tile sizes 64 and 128 are supported. Other geometry returns a Status.
Status PrepareFlashAttention(ExecutionContext& ctx, const Tensor& qo_indptr, Tensor& plan,
                             int group, int tiles, int tile_rows = 64);
Status FlashPagedAttention(ExecutionContext& ctx, const Tensor& q, const Tensor& qo,
                           const Tensor& kv, const Tensor& indices, const Tensor& last_page_len,
                           const Tensor& key, const Tensor& value, int64_t block_size,
                           const AttentionParams& params, const Tensor& plan, int tiles,
                           Tensor& out, const FlashDecodeWorkspace* decode = nullptr,
                           int tile_rows = 64);
}  // namespace inferx::ops
