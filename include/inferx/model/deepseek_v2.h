#pragma once

#include <cstdint>
#include <memory>
#include <string_view>
#include <vector>

#include "inferx/core/kv_cache.h"
#include "inferx/core/status.h"
#include "inferx/model/config.h"
#include "inferx/model/forward_batch.h"

namespace inferx::model {

/// \brief DeepSeek-V2, the first served composition of the M9 layers.
///
/// The stack is embedding → N × (RmsNorm → MLA attention → RmsNorm → FFN) →
/// RmsNorm → lm_head, where attention is `MlaAttentionLayer` over the paged
/// *latent* cache (`KvLayout{entries=1, heads=1, head_dim=kv_lora_rank +
/// qk_rope_head_dim}` — 576 wide for V2-Lite against GQA's 8192), and the FFN
/// is dense for the first `first_k_dense_replace` layers and `MoeFfn` with
/// DeepSeek's conventions (ungated shared experts, softmax/greedy routing)
/// after that. V2-Lite's `q_lora_rank: null` means Q is projected in one step;
/// the class supports both that and the full-V2 down/up form.
///
/// **Correct rather than fast, deliberately.** Attention is the unabsorbed MLA
/// form and the batch is served as a per-sequence loop: MLA's reconstruction
/// GEMM is per sequence (its `m` is that sequence's context length), so the
/// ragged batch cannot feed one kernel until either absorption or the
/// FlashInfer MLA wrapper lands (§18.7 D6). Consequences accepted for
/// bring-up:
///
///   * every decode step re-projects each sequence's whole cached context
///     through `kv_b` — O(context) work per layer per step;
///   * `CaptureDecodeGraph` is Unimplemented: the unabsorbed scratch shapes
///     depend on context length, which changes every step, so there is no
///     fixed shape to record. The engine skips capture for this model rather
///     than treating that as an error.
///
/// What this class is for is closing M9's open item: MLA wired into a `Model`,
/// serving a real checkpoint, with the RoPE convention finally settled against
/// real weights instead of asserted self-consistent.
class DeepseekV2Model {
 public:
  /// \brief Loads from a checkpoint directory.
  ///
  /// Everything goes to the device once: attention projections per layer, the
  /// per-expert FFN weights stacked into `MoeFfn`'s `[E, …]` layout through a
  /// host staging buffer (one upload per stacked tensor, not one per expert),
  /// and the fused shared-expert MLP. V2-Lite is ~31 GB bf16 — whether that
  /// fits is the caller's capacity planning (§18.6), not this loader's.
  static StatusOr<DeepseekV2Model> Load(std::string_view dir);

  ~DeepseekV2Model();
  DeepseekV2Model(const DeepseekV2Model&) = delete;
  DeepseekV2Model& operator=(const DeepseekV2Model&) = delete;
  DeepseekV2Model(DeepseekV2Model&&) noexcept;
  DeepseekV2Model& operator=(DeepseekV2Model&&) noexcept;

  const ModelConfig& config() const;

  /// \brief Runs the stack over one prompt and returns every position's logits.
  ///
  /// The logits-comparison harness, like `GptOssModel::Forward`. MLA attention
  /// needs a latent cache even for a single full-recompute pass, so this
  /// builds a temporary pool sized for the prompt, prefills through the same
  /// paged path `Step` uses, and discards it — the decode-equals-prefill pin
  /// is what makes that equivalent to a cacheless recompute. Not for use
  /// concurrently with `Step`.
  ///
  /// \param token_ids  The prompt. Positions are `0..n-1`.
  /// \param out_logits Receives `[tokens × vocab]` fp32, row-major.
  Status Forward(const std::vector<int32_t>& token_ids,
                 std::vector<float>* out_logits);

  /// \brief Allocates the paged latent cache. Required before `Step`.
  ///
  /// The layout comes from `MlaAttentionLayer::LayoutFor` — one 576-wide entry
  /// per token per layer, `ValueCache()` deliberately failing. Capacity
  /// planning note: a block holds `block_size` *latents*, ~14× more context
  /// per byte than the GQA models, and the latent replicates rather than
  /// shards under TP.
  Status AttachKvCache(int64_t num_blocks, int64_t block_size = 16);

  /// \brief The pool, for the scheduler to allocate blocks from.
  KvBlockPool* kv_pool();

  /// \brief Runs one batched step, reading and writing the paged latent cache.
  ///
  /// `batch` may hold prefill chunks, decode steps, or both. Tokens must be
  /// grouped by sequence (the scheduler's layout); the per-sequence MLA loop
  /// slices the flat batch by those groups. Synchronizes before returning.
  ///
  /// \param batch      What to run. \see ForwardBatch.
  /// \param out_logits Receives `[logits_indices.size() × vocab]` fp32,
  ///                   row-major, in the order `logits_indices` lists.
  Status Step(const ForwardBatch& batch, std::vector<float>* out_logits);

  /// \brief Sizes the activation scratch for a batch of `max_tokens`.
  Status ReserveActivations(int64_t max_tokens);

  /// \brief Unimplemented: the unabsorbed MLA path has no fixed decode shape.
  ///
  /// See the class comment. Callers that capture graphs for other models must
  /// treat Unimplemented from here as "skip", not as a failure.
  Status CaptureDecodeGraph(int64_t num_seqs, int64_t max_blocks_per_seq);

  int64_t captured_graphs() const;

 private:
  struct Impl;
  explicit DeepseekV2Model(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};

}  // namespace inferx::model
