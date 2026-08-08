#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include "inferx/core/dtype.h"
#include "inferx/core/status.h"

namespace inferx::model {

/// \brief The architecture family a checkpoint declares.
///
/// Qwen2 and Llama differ in exactly one thing that reaches the forward pass --
/// Qwen2 puts a bias on the Q/K/V projections and Llama does not -- so they are
/// one code path with a flag rather than two implementations. The enum exists
/// to reject architectures we have not read the definition of, rather than
/// running them as Llama and producing plausible nonsense.
enum class Architecture {
  kQwen2,
  kLlama,
  /// Qwen2 attention with a mixture-of-experts FFN: a router, `num_experts`
  /// expert FFNs of which `num_experts_per_tok` run per token, and a shared
  /// expert that runs for every token. Attention is bit-for-bit the Qwen2 path.
  kQwen2Moe,
  /// gpt-oss: MoE, plus four things nothing else here does — per-head learned
  /// attention **sinks**, **sliding-window** attention on alternating layers,
  /// **YaRN** position scaling, and a clamped `(up+1)·gate·σ(αgate)` activation.
  /// Its expert weights are MXFP4 (\see kernels/mxfp4.h).
  kGptOss,
  /// DeepSeek-V2: **MLA** attention over a compressed latent cache, and
  /// DeepSeekMoE — routed experts behind the first `first_k_dense_replace`
  /// dense layers, plus **ungated** always-on shared experts (Qwen2-MoE gates
  /// its shared expert; DeepSeek adds it unconditionally). YaRN scaling with
  /// the `mscale` softmax-scale correction. V2-Lite additionally has no Q
  /// down-projection (`q_lora_rank` null).
  kDeepSeekV2,
};

const char* ArchitectureName(Architecture arch);

/// \brief A decoder-only transformer's shape, as read from `config.json`.
///
/// Every field is taken from the file; nothing is inferred except `head_dim`,
/// which HF omits when it is `hidden_size / num_attention_heads` and states
/// explicitly when it is not.
struct ModelConfig {
  Architecture architecture = Architecture::kQwen2;

  int64_t hidden_size = 0;
  int64_t intermediate_size = 0;
  int64_t num_hidden_layers = 0;
  int64_t num_attention_heads = 0;
  /// Fewer than `num_attention_heads` for GQA; equal for MHA.
  int64_t num_key_value_heads = 0;
  int64_t head_dim = 0;
  int64_t vocab_size = 0;
  int64_t max_position_embeddings = 0;

  double rms_norm_eps = 1e-6;
  double rope_theta = 10000.0;

  /// When true there is no `lm_head.weight`; the output projection reuses the
  /// embedding matrix. Qwen2.5-3B sets this.
  bool tie_word_embeddings = false;

  /// Qwen2 biases Q/K/V (never O); Llama biases nothing.
  bool attention_bias = false;

  /// The dtype the checkpoint stores, from `torch_dtype`. Usually BF16.
  DataType weight_dtype = DataType::kBFloat16;

  // --- Mixture of experts (§7.3). Zero experts means a dense FFN, which is
  // what every field below degenerates to and why they need no separate flag.

  /// Routed experts per layer. 0 for a dense model.
  int64_t num_experts = 0;

  /// Experts each token is routed to. The router picks this many of
  /// `num_experts` by gate score; the rest cost nothing for that token.
  int64_t num_experts_per_tok = 0;

  /// The FFN width *of one routed expert*, which is far narrower than a dense
  /// model's `intermediate_size` -- that is what makes a wide expert count
  /// affordable. Falls back to `intermediate_size` when a checkpoint omits it.
  int64_t moe_intermediate_size = 0;

  /// Rescale the top-k gate weights to sum to 1. Qwen2-MoE and Mixtral do;
  /// some architectures deliberately do not, so it is read rather than assumed.
  bool norm_topk_prob = true;

  /// Width of the always-on shared expert, which every token passes through in
  /// addition to its routed ones. Qwen2-MoE gates it by its own sigmoid;
  /// DeepSeek adds it ungated — the convention follows the architecture, not a
  /// config key. 0 means none. DeepSeek spells this as a *count*
  /// (`n_shared_experts`); parsing multiplies by `moe_intermediate_size`.
  int64_t shared_expert_intermediate_size = 0;

  /// The first `first_k_dense_replace` layers use a dense FFN of
  /// `intermediate_size` even in a MoE model. DeepSeek-V2 sets 1; every other
  /// MoE checkpoint here sets 0.
  int64_t first_k_dense_replace = 0;

  /// A layer past `first_k_dense_replace` is MoE when its index divides by
  /// this. 1 — every checkpoint we serve — means all of them.
  int64_t moe_layer_freq = 1;

  /// Multiplies the routed experts' combined output. 1.0 everywhere except
  /// some DeepSeek configs; read so those are not silently mis-scaled.
  double routed_scaling_factor = 1.0;

  /// \brief Whether the FFN is a mixture of experts.
  bool is_moe() const { return num_experts > 0; }

  /// \brief Whether layer `i`'s FFN is routed rather than dense.
  ///
  /// HF's DeepSeek reference gates on `layer >= first_k_dense_replace &&
  /// layer % moe_layer_freq == 0`; this mirrors it exactly, including the
  /// modulus being over the absolute index rather than the post-dense offset.
  bool IsMoeLayer(int64_t layer) const {
    if (!is_moe()) return false;
    if (layer < 0 || layer >= num_hidden_layers) return false;
    if (layer < first_k_dense_replace) return false;
    return moe_layer_freq > 0 && layer % moe_layer_freq == 0;
  }

  // --- Multi-head latent attention (§7.3). Zero `kv_lora_rank` means ordinary
  // GQA, which is what every field below degenerates to.

  /// Width of the compressed KV latent — the thing MLA caches *instead of*
  /// per-head K and V. 0 for a GQA model.
  int64_t kv_lora_rank = 0;

  /// Width of the Q down-projection. 0 means Q is projected from the hidden
  /// state in one step, which is DeepSeek-V2-Lite's shape.
  int64_t q_lora_rank = 0;

  /// The part of each Q/K head that carries no position information and is
  /// reconstructed from the latent.
  int64_t qk_nope_head_dim = 0;

  /// The part that carries RoPE. Decoupled: it is cached once per token rather
  /// than once per head, which is what keeps the latent small.
  int64_t qk_rope_head_dim = 0;

  /// Width of each V head, which MLA lets differ from the Q/K head width.
  int64_t v_head_dim = 0;

  /// \brief Whether attention is MLA rather than GQA.
  bool is_mla() const { return kv_lora_rank > 0; }

  // --- gpt-oss (§14, M11). All default to "off", which is every other model.

  /// Per-head learned logits in the softmax denominator with no value behind
  /// them. True when the checkpoint carries `self_attn.sinks`.
  bool attention_sinks = false;

  /// Width of the sliding attention window, 0 when every layer is full.
  int64_t sliding_window = 0;

  /// One entry per layer: true where attention only sees the last
  /// `sliding_window` tokens. Empty means every layer is full attention.
  ///
  /// A vector rather than a stride because "every other layer" is a property
  /// of this checkpoint rather than of the idea, and a model that alternated
  /// differently would otherwise be silently mis-run.
  std::vector<bool> layer_is_sliding;

  /// The clamp in gpt-oss's gated activation. 0 means the ordinary SwiGLU.
  double swiglu_limit = 0.0;

  /// `x·σ(αx)` rather than `x·σ(x)`; 1.702 makes it approximate GELU.
  double swiglu_alpha = 1.702;

  // --- YaRN position scaling. `yarn_factor` 0 means no scaling.

  double yarn_factor = 0.0;
  double yarn_beta_fast = 32.0;
  double yarn_beta_slow = 1.0;
  int64_t yarn_original_max_position = 0;
  bool yarn_truncate = true;

  /// DeepSeek's YaRN mscale pair. The attention softmax scale is multiplied by
  /// `m(mscale)² / m(mscale_all_dim)`-style corrections where
  /// `m(x) = 0.1·x·ln(factor) + 1`; when the two are equal (V2-Lite: both
  /// 0.707) the cos/sin factor cancels to 1 and the whole effect lands in the
  /// softmax scale. HF defaults: mscale 1, mscale_all_dim 0.
  double yarn_mscale = 1.0;
  double yarn_mscale_all_dim = 0.0;

  /// \brief Whether positions are YaRN-scaled.
  bool is_yarn() const { return yarn_factor > 0.0; }

  /// \brief Whether layer `i` attends only within the sliding window.
  bool IsSlidingLayer(int64_t layer) const {
    if (sliding_window <= 0) return false;
    if (layer_is_sliding.empty()) return false;
    if (layer < 0 || layer >= static_cast<int64_t>(layer_is_sliding.size())) {
      return false;
    }
    return layer_is_sliding[static_cast<size_t>(layer)];
  }

  /// \brief Bytes one token occupies in one layer's KV cache.
  ///
  /// A property of the model, never a formula the scheduler applies (§7.3, T11).
  /// For GQA it is `2 · kv_heads · head_dim`; for MLA it is the latent plus the
  /// one shared RoPE key, with no factor of two and no head count — and, the
  /// part that catches people, **it does not shrink under tensor parallelism**,
  /// because the latent is replicated across ranks rather than sharded.
  int64_t KvElementsPerTokenPerLayer() const {
    if (is_mla()) return kv_lora_rank + qk_rope_head_dim;
    return 2 * num_key_value_heads * head_dim;
  }

  /// \brief Q heads per KV head. 1 means MHA.
  int64_t gqa_group_size() const {
    return num_key_value_heads > 0 ? num_attention_heads / num_key_value_heads
                                   : 0;
  }

  /// \brief Total width of the Q projection. **GQA only** — under MLA the
  /// derived `head_dim` is meaningless (the real QK head is
  /// `qk_nope_head_dim + qk_rope_head_dim` wide) and this must not be
  /// consulted.
  int64_t q_dim() const { return num_attention_heads * head_dim; }

  /// \brief Total width of each of the K and V projections. **GQA only**, as
  /// with `q_dim()`.
  int64_t kv_dim() const { return num_key_value_heads * head_dim; }

  /// \brief Checks the fields against each other.
  ///
  /// Separate from parsing because a config can be syntactically fine and still
  /// describe something impossible -- heads that do not divide, a GQA ratio
  /// that is not integral. Those are worth catching before the first weight is
  /// read rather than as a shape mismatch 36 layers in.
  Status Validate() const;

  std::string ToString() const;

  /// \brief Parses a `config.json` document.
  ///
  /// \param json_text The file's contents.
  /// \return          The config, already validated, or the first problem.
  static StatusOr<ModelConfig> FromJson(std::string_view json_text);

  /// \brief Reads and parses `<dir>/config.json`.
  static StatusOr<ModelConfig> FromDirectory(std::string_view dir);
};

}  // namespace inferx::model
