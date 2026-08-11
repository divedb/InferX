#pragma once

#include <cstdint>
#include <memory>
#include <string_view>
#include <vector>

#include "inferx/core/kv_cache.h"
#include "inferx/core/status.h"
#include "inferx/model/config.h"
#include "inferx/model/forward_batch.h"
#include "inferx/model/safetensors.h"

namespace inferx::model {

/// \brief gpt-oss-20b, forward only, correct rather than fast.
///
/// M11 Phase 2's deliverable, and it is deliberately the same kind of object
/// M2's `Qwen2Model::Forward` was: batch 1, no KV cache, full recompute of
/// every position on every call, synchronizing before it returns. It exists so
/// that its logits can be compared against HuggingFace's, which is the only way
/// to know that the MXFP4 decode, the routing, the sinks, the sliding window,
/// YaRN and the clamped activation are all right *together* rather than
/// individually.
///
/// **The memory strategy is the interesting part.** gpt-oss's expert weights
/// are 10.1 GB as MXFP4 and ~38 GB dequantized, so they cannot all be bf16 on
/// a 16 GB card — and this class does not have a 4-bit GEMM to avoid that with.
/// So the experts stay **packed on the host**, and each layer is uploaded and
/// dequantized into one reusable 1.6 GB scratch as it is reached:
///
///     per layer: upload 423 MB of MXFP4  ->  dequantize to 1.59 GB bf16
///                run the layer            ->  reuse the same scratch
///
/// Peak device usage is about 5.5 GB, and a forward pass moves ~10 GB over
/// PCIe. That is roughly a second per call and completely acceptable for
/// something run once against a reference. It is *not* a serving path and must
/// not be mistaken for one: Phase 4's job is a mainloop that reads MXFP4
/// directly, at which point this class stays exactly as it is, as the thing
/// that says the fast path is right.
class GptOssModel {
 public:
  /// \brief Loads from a checkpoint directory.
  ///
  /// The non-expert weights (attention, norms, router, embedding, lm_head —
  /// 3.6 GB) go to the device once. The expert weights are left in the
  /// checkpoint's mapping and read per layer per call.
  static StatusOr<GptOssModel> Load(std::string_view dir, DeviceId device);

  ~GptOssModel();
  GptOssModel(const GptOssModel&) = delete;
  GptOssModel& operator=(const GptOssModel&) = delete;
  GptOssModel(GptOssModel&&) noexcept;
  GptOssModel& operator=(GptOssModel&&) noexcept;

  const ModelConfig& config() const;

  /// \brief Runs the stack over `token_ids` and returns every position's
  /// logits.
  ///
  /// \param token_ids  The prompt. Positions are `0..n-1`.
  /// \param out_logits Receives `[tokens × vocab]` fp32, row-major. fp32
  ///                   because the caller is a comparison against a reference,
  ///                   even though the stack ran in bf16.
  Status Forward(const std::vector<int32_t>& token_ids,
                 std::vector<float>* out_logits);

  /// \brief Allocates the paged KV cache. Required before `Step`.
  ///
  /// The Phase 3 slow path (Forward) keeps no cache and recomputes every
  /// position; the paged path (Step) writes each layer's K/V into this pool
  /// and reads it back, which is what lets the scheduler batch multiple
  /// sequences and what amortises the MXFP4 expert upload across a batch. The
  /// 12 sliding-attention layers are served as full-attention for now (they
  /// cache every token rather than 128) -- a memory cost, not a correctness
  /// one, deferred until R-C's per-layer-lifetime scheduler change.
  Status AttachKvCache(int64_t num_blocks, int64_t block_size = 16);

  /// \brief Bytes one KV block would occupy across all layers, without
  /// allocating. For sizing the pool before `AttachKvCache`.
  int64_t KvBlockBytes(int64_t block_size) const;

  /// \brief The pool, for the scheduler to allocate blocks from. See
  /// Qwen2Model.
  KvBlockPool* kv_pool();

  /// \brief Runs one batched step, reading and writing the paged KV cache.
  ///
  /// The continuous-batching counterpart of `Forward`: `batch` may hold a
  /// prefill, a set of decode steps, or both, across multiple sequences. The
  /// expert dequantize-and-forward path is unchanged from `Forward` -- it
  /// already processes whatever tokens it is handed -- so a batch of N tokens
  /// pays one ~10 GB PCIe expert upload rather than N, which is the whole
  /// reason batching helps gpt-oss before Phase 4's fused GEMM lands.
  ///
  /// Attention runs through our own `PagedAttentionWithLse` (the reference
  /// kernel, with the lse output and per-layer sliding window gpt-oss needs)
  /// rather than FlashInfer, because the sink rescale needs the lse and the
  /// FlashInfer wrapper does not expose it yet. Slower per launch; attention is
  /// not the bottleneck.
  ///
  /// Synchronizes before returning and argmaxes on the host, so this remains
  /// the synchronous sibling of Qwen2Model::Step. Fixed decode shapes may be
  /// CUDA-graph replayed; prefill remains launch-by-launch.
  ///
  /// \param batch      What to run. \see ForwardBatch.
  /// \param out_logits Receives `[logits_indices.size() × vocab]` fp32,
  ///                   row-major, in the order `logits_indices` lists.
  Status Step(const ForwardBatch& batch, std::vector<float>* out_logits);

  /// \brief Sizes the activation scratch for a batch of `max_tokens`.
  ///
  /// Same correctness contract as Qwen2Model::ReserveActivations: the scratch
  /// only grows, so reserving the largest expected batch up front keeps every
  /// later growth a no-op. Required before any `Step` larger than what
  /// `Forward` already sized.
  Status ReserveActivations(int64_t max_tokens);

  /// \brief Captures one fixed decode shape for replay by `Step`.
  ///
  /// Decode has one token per sequence. The graph records the device body only;
  /// `Step` still uploads positions, slots and block tables before each replay
  /// and downloads the requested logits afterwards.
  Status CaptureDecodeGraph(int64_t num_seqs, int64_t max_blocks_per_seq);

  int64_t captured_graphs() const;

 private:
  struct Impl;
  explicit GptOssModel(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};

}  // namespace inferx::model
