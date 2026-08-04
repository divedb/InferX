#pragma once

#include <cstddef>
#include <memory>
#include <vector>

#include <cuda_runtime_api.h>

#include "absl/types/span.h"
#include "inferx/core/status.h"
#include "inferx/core/tensor_view.h"

namespace inferx::kernels {

/// \brief FlashInfer's batched paged decode attention, wrapped for our types.
///
/// The performance path that replaces `PagedAttention`. Our own kernel stays as
/// the reference it is checked against (R5): one is written to be obviously
/// correct, the other to be fast, and the only way that stays honest is a test
/// that runs both.
///
/// What is used here is FlashInfer's **kernel template layer** and nothing
/// else. §9 notes that its wrapper layer is increasingly torch- and
/// JIT-coupled while the templates are not, and that turns out to be exactly
/// true: `include/flashinfer/` has no torch include anywhere in it. So this
/// compiles the templates ahead of time against our own `TensorView`, which is
/// the trade §9 describes -- upgrades become manual merges rather than version
/// bumps, and the pin is by commit.
///
/// One structural difference is worth knowing before using this. FlashInfer
/// describes a batch's pages in CSR form -- a flat `indices` array with a
/// per-sequence `indptr` -- where our block table is a dense
/// `[seqs, max_blocks]` grid. The dense form is what a CUDA-graph-captured step
/// wants (§6.2: a fixed buffer, no re-capture), so it stays, and this class
/// takes the CSR form the kernel needs. `BuildCsrBlockTable` converts.
class FlashInferDecode {
 public:
  /// FlashInfer partitions long sequences across blocks and reduces afterwards,
  /// which needs scratch proportional to the split. These are its own defaults;
  /// too small shows up as a plan failure rather than a wrong answer.
  static constexpr size_t kDefaultFloatWorkspaceBytes = 128u << 20;
  static constexpr size_t kDefaultIntWorkspaceBytes = 8u << 20;

  /// \brief Allocates the workspaces. One instance per executor rank.
  static StatusOr<FlashInferDecode> Create(
      size_t float_workspace_bytes = kDefaultFloatWorkspaceBytes,
      size_t int_workspace_bytes = kDefaultIntWorkspaceBytes);

  ~FlashInferDecode();
  FlashInferDecode(const FlashInferDecode&) = delete;
  FlashInferDecode& operator=(const FlashInferDecode&) = delete;
  FlashInferDecode(FlashInferDecode&&) noexcept;
  FlashInferDecode& operator=(FlashInferDecode&&) noexcept;

  /// \brief One decode step: exactly one query token per sequence.
  ///
  /// Prefill is not served here. FlashInfer has a separate ragged-prefill
  /// kernel for that, and decode is where the win is -- it is the shape the
  /// engine spends its life in and the one our naive kernel is worst at, since
  /// that one walks every key with a single block.
  ///
  /// \param q             `[batch, q_heads, head_dim]` bf16, one row per
  ///                      sequence.
  /// \param k_cache       `[blocks, block_size, kv_heads, head_dim]` bf16.
  ///                      This is FlashInfer's NHD layout unchanged, which is
  ///                      why no repacking is needed.
  /// \param v_cache       Same shape.
  /// \param kv_indices    Flat page indices for the whole batch, concatenated
  ///                      in sequence order.
  /// \param kv_indptr     `[batch + 1]` int32 **on the device**, prefix sums
  ///                      into `kv_indices`.
  /// \param kv_indptr_host The same `batch + 1` values **on the host**.
  ///
  ///   Both are required, and passing the host copy rather than reading it back
  ///   is the whole reason this parameter exists. FlashInfer's planner needs the
  ///   indptr host-side to estimate how to split the work, and an earlier
  ///   version of this wrapper obtained it with a device-to-host copy followed
  ///   by `cudaStreamSynchronize` -- a full device synchronization inside the
  ///   decode loop, which §5.2 forbids outright and which would have made M6's
  ///   overlap pipeline impossible.
  ///
  ///   The copy was never necessary: the scheduler *computes* this array while
  ///   walking its own block tables (§3.1), so the host already holds it before
  ///   anything reaches the device. Round-tripping it was the bug.
  ///
  /// \param last_page_len `[batch]` int32, tokens used in each sequence's final
  ///                      page. 1..block_size, never 0.
  /// \param out           `[batch, q_heads, head_dim]` bf16.
  /// \param scale         Usually `1/sqrt(head_dim)`.
  /// \param stream        Stream to launch on. Does **not** synchronize, and
  ///                      must not: see above.
  Status Decode(const TensorView& q, const TensorView& k_cache,
                const TensorView& v_cache, const TensorView& kv_indices,
                const TensorView& kv_indptr,
                absl::Span<const int32_t> kv_indptr_host,
                const TensorView& last_page_len, const TensorView& out,
                float scale, cudaStream_t stream = nullptr);

  /// \brief One decode step against an **FP8 e4m3** paged KV cache.
  ///
  /// Same shape contract as `Decode`, but the cache holds K/V as fp8 e4m3
  /// rather than bf16 -- half the KV memory (§6.4), which is the capacity win.
  /// Q and O stay bf16; FlashInfer's fa2 decode upcasts fp8 to fp32 in
  /// registers on load, so this uses no fp8 tensor cores and runs on sm_89,
  /// unlike the Hopper-only mixed-input GEMM.
  ///
  /// FP8 is a scaled format, so two per-tensor dequant scalars are required,
  /// and where they go is fixed by the attention math rather than chosen:
  /// `k_scale` is folded into `scale` (attention is invariant to a rescaling
  /// of K -- it falls out of the normalizing softmax), and `v_scale` is
  /// applied to `out` after the kernel returns. That is exactly what
  /// FlashInfer's own Python wrapper does, and the TODO there to fuse `v_scale`
  /// into the kernel is still open, so neither happens inside the template.
  ///
  /// This is the one-shot (non-graph) entry point: it plans and launches in one
  /// call, like `Decode`. A graph-capturable FP8 path (Plan/Run analogues) is
  /// left for serving-path integration.
  Status DecodeFp8(const TensorView& q, const TensorView& k_cache,
                   const TensorView& v_cache, const TensorView& kv_indices,
                   const TensorView& kv_indptr,
                   absl::Span<const int32_t> kv_indptr_host,
                   const TensorView& last_page_len, const TensorView& out,
                   float scale, float k_scale, float v_scale,
                   cudaStream_t stream = nullptr);

  /// \name Graph-capturable FP8 KV decode
  ///
  /// The Plan/Run split for the FP8 KV path, mirroring the bf16 `Plan`/`Run`.
  /// `PlanFp8` does the host-side work (reads `kv_indptr_host`, branches on it)
  /// and must stay outside a captured region; `RunFp8` is pure device work and
  /// can be recorded. The scales are fixed at capture time, which is consistent
  /// with a per-layer fixed KV scale -- the serving strategy M8 will use.
  ///@{
  Status PlanFp8(int64_t batch, int64_t q_heads, int64_t kv_heads,
                 int64_t head_dim, int64_t page_size,
                 absl::Span<const int32_t> kv_indptr_host, bool graph_safe,
                 cudaStream_t stream = nullptr);

  Status RunFp8(const TensorView& q, const TensorView& k_cache,
                const TensorView& v_cache, const TensorView& kv_indices,
                const TensorView& kv_indptr, const TensorView& last_page_len,
                const TensorView& out, float scale, float k_scale,
                float v_scale, cudaStream_t stream = nullptr);
  ///@}

  /// \brief Plans a step without launching it. Host-side; **not capturable**.
  ///
  /// The split that lets FlashInfer live inside a CUDA graph. Planning reads
  /// the indptr on the host and branches on it, so it can never be recorded;
  /// launching is pure device work and can. Calling `Plan` once per step
  /// outside the captured region and `Run` inside it gives a graph that
  /// contains FlashInfer -- which is the only way to reach the launch overhead
  /// that dominates a small-batch decode step.
  ///
  /// \param graph_safe When true, plans in FlashInfer's fixed-shape mode:
  ///   `padded_batch_size` becomes a function of the device and head count
  ///   rather than of the batch's sequence lengths, and the workspace offsets
  ///   it hands back stop moving between steps. That is precisely the property
  ///   a graph needs -- the structure is pinned while the values stay live --
  ///   and it is why a graph captured at one context length keeps working as
  ///   sequences grow. It forces the split-KV path on unconditionally, which
  ///   costs a reduction pass on short sequences.
  ///
  /// \param batch, q_heads, kv_heads, head_dim, page_size  The shape.
  /// \param kv_indptr_host `[batch + 1]` prefix sums, on the host.
  Status Plan(int64_t batch, int64_t q_heads, int64_t kv_heads,
              int64_t head_dim, int64_t page_size,
              absl::Span<const int32_t> kv_indptr_host, bool graph_safe,
              cudaStream_t stream = nullptr);

  /// \brief Launches the plan from the last `Plan` call. Capturable.
  ///
  /// Every pointer it passes is stable across steps and every value it reads
  /// lives in device memory that `Plan` rewrote, so a recording of this call
  /// replays correctly for any later step of the same shape.
  ///
  /// The tensors must match the shape last planned; mismatches are rejected
  /// rather than launched, because a plan built for a different batch produces
  /// a kernel reading past the end of its indices.
  Status Run(const TensorView& q, const TensorView& k_cache,
             const TensorView& v_cache, const TensorView& kv_indices,
             const TensorView& kv_indptr, const TensorView& last_page_len,
             const TensorView& out, float scale,
             cudaStream_t stream = nullptr);

 private:
  struct Impl;

  explicit FlashInferDecode(std::unique_ptr<Impl> impl);

  std::unique_ptr<Impl> impl_;
};

/// \brief Converts a dense block table to the CSR form FlashInfer wants.
///
/// \param block_table   Row-major `[num_seqs, max_blocks_per_seq]`.
/// \param blocks_used   `[num_seqs]`, how many blocks each sequence actually
///                      holds. The rest of its row is padding.
/// \param out_indices   Receives the concatenated page indices.
/// \param out_indptr    Receives `num_seqs + 1` prefix sums.
Status BuildCsrBlockTable(const std::vector<int32_t>& block_table,
                          int64_t num_seqs, int64_t max_blocks_per_seq,
                          const std::vector<int32_t>& blocks_used,
                          std::vector<int32_t>* out_indices,
                          std::vector<int32_t>* out_indptr);

}  // namespace inferx::kernels
