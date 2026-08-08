#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "inferx/core/kv_cache.h"
#include "inferx/core/status.h"
#include "inferx/core/tensor_view.h"
#include "inferx/kernels/gemm.h"
#include "inferx/model/config.h"

namespace inferx::model {

/// \brief The weights of one multi-head latent attention layer.
///
/// Named after DeepSeek-V2's checkpoint, because that is what a loader will be
/// matching against and a rename here would only move the translation
/// somewhere less obvious.
struct MlaWeights {
  /// `[q_lora_rank, hidden]`. Undefined when `q_lora_rank` is 0, in which case
  /// `q_b` projects straight from the hidden state.
  TensorView q_a;
  /// `[q_lora_rank]` — RMSNorm between the two Q projections.
  TensorView q_a_norm;
  /// `[heads · (qk_nope + qk_rope), q_lora_rank or hidden]`.
  TensorView q_b;

  /// `[kv_lora_rank + qk_rope, hidden]` — the down-projection that produces
  /// both the latent and the shared RoPE key in one GEMM. "with_mqa" in the
  /// checkpoint, because the RoPE key is one head shared by all of them.
  TensorView kv_a;
  /// `[kv_lora_rank]` — RMSNorm applied to the latent before it is cached, so
  /// what the cache holds is already normalized.
  TensorView kv_a_norm;
  /// `[heads · (qk_nope + v_head_dim), kv_lora_rank]` — reconstructs each
  /// head's K and V from the latent.
  TensorView kv_b;

  /// `[hidden, heads · v_head_dim]`.
  TensorView o;
};

/// \brief Multi-head latent attention over a paged latent cache.
///
/// What MLA changes, relative to the GQA path in `Qwen2Model`, is one thing
/// with consequences everywhere: it caches a **compressed latent per token**
/// and reconstructs each head's K and V from it, instead of caching K and V
/// per head. For DeepSeek-V2's shape that is 576 elements per token per layer
/// against 8192 — the reason MLA exists, and the reason §7.3 insisted the KV
/// pool be parameterized by `KvLayout` before there was anything to
/// parameterize it for.
///
/// The pool is instantiated with `entries_per_token = 1`, `kv_heads = 1`,
/// `head_dim = kv_lora_rank + qk_rope_head_dim`, and the whole latent is read
/// through `KeyCache()`. `ValueCache()` fails for this layout, deliberately:
/// there is no separate V to return.
///
/// **This is the unabsorbed form.** Every step reconstructs K and V for the
/// whole context with one `kv_b` GEMM, then runs ordinary attention. That is
/// correct and it is what the tests are written against, but it gives back
/// most of what MLA is for at decode time: the reconstruction is O(context)
/// work per step, where the absorbed form folds `kv_b` into the query once and
/// attends against the latent directly, making a decode step O(context) in the
/// *latent* width instead. Absorption is the next step and it changes no
/// interface here — see §14, M9.
///
/// Tensor parallelism is not implemented, and the shape MLA forces is worth
/// stating because it is the opposite of GQA's: the latent is **replicated**
/// across ranks, not sharded, since every head's reconstruction needs all of
/// it. KV memory therefore does not fall with `tp_size` for an MLA model,
/// which is why `ModelConfig::KvElementsPerTokenPerLayer()` is a model
/// property and never a scheduler formula (T11).
class MlaAttentionLayer {
 public:
  /// \brief The KV geometry an MLA model needs from the pool.
  ///
  /// Handed to `KvBlockPool::Create` in place of the GQA layout. A model that
  /// computed this itself would be the T11 mistake.
  static KvLayout LayoutFor(const ModelConfig& config);

  /// \brief Builds the layer and sizes its scratch.
  ///
  /// YaRN is resolved here, once, from the config: the blended frequency table
  /// for the rope sub-vector goes to the device, and the softmax scale becomes
  /// `YarnMscale(factor, mscale_all_dim)² / sqrt(qk_nope + qk_rope)` — for
  /// DeepSeek-V2-Lite ≈ 0.11472 against the unscaled 0.07217. A config without
  /// YaRN keeps the closed-form rope and the plain `1/sqrt` scale.
  ///
  /// \param max_tokens   Largest query batch that will ever be passed.
  /// \param max_context  Longest context that will ever be attended over. The
  ///                     reconstruction buffers are O(context · heads), which
  ///                     is the unabsorbed form's memory cost and another
  ///                     reason absorption is worth doing.
  static StatusOr<MlaAttentionLayer> Create(const ModelConfig& config,
                                            int64_t max_tokens,
                                            int64_t max_context);

  /// \brief The attention softmax scale in use, YaRN correction included.
  ///
  /// Exposed so a test can pin the DeepSeek constant and a model can report
  /// it; `Forward` already applies it.
  float softmax_scale() const;

  ~MlaAttentionLayer();
  MlaAttentionLayer(const MlaAttentionLayer&) = delete;
  MlaAttentionLayer& operator=(const MlaAttentionLayer&) = delete;
  MlaAttentionLayer(MlaAttentionLayer&&) noexcept;
  MlaAttentionLayer& operator=(MlaAttentionLayer&&) noexcept;

  /// \brief Runs one sequence's attention, appending to and reading from the
  ///        latent cache.
  ///
  /// One sequence rather than a batch: MLA's reconstruction GEMM is per
  /// sequence (its `m` is that sequence's context length), so batching means
  /// either a grouped GEMM or a loop, and the loop belongs in the caller where
  /// the batch is known. The GQA path's ragged-batch handling has no analogue
  /// here yet.
  ///
  /// \param x            `[tokens, hidden]` bf16 — the layer's input, normed.
  /// \param positions    `[tokens]` int32, absolute positions.
  /// \param slot_mapping `[tokens]` int32, cache slots for the new tokens.
  /// \param block_table  `[blocks]` int32, the sequence's blocks in order.
  /// \param context_len  Tokens in the cache *after* this call's append —
  ///                     i.e. `past + tokens`.
  /// \param cache        `pool.KeyCache(layer)`.
  /// \param out          `[tokens, hidden]` bf16.
  Status Forward(const TensorView& x, const TensorView& positions,
                 const TensorView& slot_mapping, const TensorView& block_table,
                 int64_t context_len, const TensorView& cache,
                 const MlaWeights& weights, const TensorView& out,
                 kernels::CublasLtGemm* gemm, cudaStream_t stream = nullptr);

 private:
  struct Impl;
  explicit MlaAttentionLayer(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};

}  // namespace inferx::model
