#pragma once

#include <cstdint>
#include <memory>

#include <cuda_runtime_api.h>

#include "absl/types/span.h"
#include "inferx/core/status.h"
#include "inferx/core/tensor_view.h"

namespace inferx::kernels {

/// \brief FlashInfer's paged prefill kernel: many query tokens per sequence.
///
/// The decode wrapper next door serves exactly one query token per sequence,
/// which is why every prompt currently falls through to the M2 reference
/// kernel. That kernel stages the whole key sequence in shared memory, so it
/// costs 2.3 s for a 2k-token prompt, 108 s for 8k, and cannot launch at all
/// past ~12k keys. Prefill is the engine's worst number by three orders of
/// magnitude and this is what replaces it.
///
/// Same split as `FlashInferDecode`, for the same reason: `Plan` reads indptr
/// arrays on the host and branches on them, so it can never be recorded into a
/// CUDA graph; `Run` is pure device work. Prefill is not currently captured --
/// prompt lengths vary per request, so a graph per shape is not worth having --
/// but keeping the split means that decision can change without reworking this.
///
/// Ragged queries are the substantive difference. Decode has one query row per
/// sequence and needs only `kv_indptr`; prefill has a variable number, so it
/// needs `qo_indptr` as well, and the planner uses both to tile work across
/// CTAs. Sequences in one batch may have different prompt lengths, which is
/// exactly the case the reference kernel handles by brute force and this one
/// handles by scheduling.
///
/// Not implemented, and rejected rather than ignored: custom masks, ALiBi,
/// sliding windows, logits soft-cap, and fused RoPE. Qwen2 applies RoPE before
/// attention and uses plain causal masking, so the only mask mode wired is
/// causal.
class FlashInferPrefill {
 public:
  /// \brief Allocates the workspaces the planner and kernel need.
  ///
  /// \param float_workspace_bytes Scratch for split-KV partial outputs. Scales
  ///                              with the number of CTAs the planner creates,
  ///                              not with the batch.
  /// \param int_workspace_bytes   Scratch for the plan's index arrays.
  static StatusOr<std::unique_ptr<FlashInferPrefill>> Create(
      size_t float_workspace_bytes = 128u * 1024 * 1024,
      size_t int_workspace_bytes = 8u * 1024 * 1024);

  ~FlashInferPrefill();

  FlashInferPrefill(const FlashInferPrefill&) = delete;
  FlashInferPrefill& operator=(const FlashInferPrefill&) = delete;

  /// \brief Plans one batch. Host-side; must not be inside a graph capture.
  ///
  /// \param qo_indptr_host `[batch + 1]` prefix sums over query rows. This is
  ///                       what makes the batch ragged: sequence i contributes
  ///                       rows `[qo_indptr[i], qo_indptr[i + 1])`.
  /// \param kv_indptr_host `[batch + 1]` prefix sums over KV *pages*.
  /// \param total_rows     `qo_indptr_host.back()`, passed explicitly because
  ///                       the planner wants it before it reads the array.
  Status Plan(int64_t batch, int64_t q_heads, int64_t kv_heads,
              int64_t head_dim, int64_t page_size,
              absl::Span<const int32_t> qo_indptr_host,
              absl::Span<const int32_t> kv_indptr_host, int64_t total_rows,
              cudaStream_t stream = nullptr);

  /// \brief Launches the plan from the last `Plan` call.
  ///
  /// \param q             `[total_rows, q_heads, head_dim]` bf16, rows ordered
  ///                      by sequence and then by position within it.
  /// \param qo_indptr     Device copy of the array given to `Plan`.
  /// \param kv_indices    Page ids, CSR, indexed by `kv_indptr`.
  /// \param last_page_len `[batch]` tokens used in each sequence's final page.
  /// \param out           `[total_rows, q_heads, head_dim]` bf16.
  Status Run(const TensorView& q, const TensorView& k_cache,
             const TensorView& v_cache, const TensorView& qo_indptr,
             const TensorView& kv_indices, const TensorView& kv_indptr,
             const TensorView& last_page_len, const TensorView& out,
             float scale, cudaStream_t stream = nullptr);

  /// \brief `Plan` followed by `Run`, for callers with no graph to worry about.
  Status Prefill(const TensorView& q, const TensorView& k_cache,
                 const TensorView& v_cache, const TensorView& qo_indptr,
                 absl::Span<const int32_t> qo_indptr_host,
                 const TensorView& kv_indices, const TensorView& kv_indptr,
                 absl::Span<const int32_t> kv_indptr_host,
                 const TensorView& last_page_len, const TensorView& out,
                 float scale, cudaStream_t stream = nullptr);

  /// \brief Same as `Prefill` against an **FP8 e4m3** paged KV cache.
  ///
  /// The prefill writes the prompt's K/V and attends over them, so for a cache
  /// that decode will read as fp8, prefill must speak fp8 too -- otherwise the
  /// prompt's K/V land in bf16 and decode reads them as fp8 (garbage). `v_scale`
  /// applies to `out` after the kernel, as in `FlashInferDecode::DecodeFp8`.
  /// `k_scale` is folded into the query (`q*k_scale`, into a scratch buffer) and
  /// `sm_scale` stays at `scale` -- not folded into `sm_scale` like decode: fa2
  /// prefill suppresses masked/padding K positions inside
  /// `exp2(logit*sm_scale_log2)`, and the `scale*k_scale` product (~1e-4) is too
  /// small to zero them, so they leak into the softmax denominator and attenuate
  /// the output. Folding into Q is algebraically identical and keeps `sm_scale`
  /// at 1/sqrt(head_dim). Decode needs no such care -- it excludes invalid slots
  /// structurally, with no soft mask to defeat. One-shot (no Plan/Run split)
  /// because prefill is not graph-captured -- prompt shapes vary.
  Status PrefillFp8(const TensorView& q, const TensorView& k_cache,
                    const TensorView& v_cache, const TensorView& qo_indptr,
                    absl::Span<const int32_t> qo_indptr_host,
                    const TensorView& kv_indices, const TensorView& kv_indptr,
                    absl::Span<const int32_t> kv_indptr_host,
                    const TensorView& last_page_len, const TensorView& out,
                    float scale, float k_scale, float v_scale,
                    cudaStream_t stream = nullptr);

 private:
  struct Impl;

  explicit FlashInferPrefill(std::unique_ptr<Impl> impl);

  std::unique_ptr<Impl> impl_;
};

}  // namespace inferx::kernels
