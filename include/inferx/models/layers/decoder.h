/// \file
/// \brief The shared pre-norm decoder stack: token embedding, homogeneous
/// or heterogeneous decoder layers, final normalization. This helper serves
/// architectures whose execution order matches (Llama lineage); models
/// remain free to compose layers differently and implement the
/// CausalLanguageModel boundary directly.
///
/// Phase-agnostic: prefill and decode are expressed entirely in the
/// attention batch's ragged geometry.

#ifndef INFERX_MODELS_LAYERS_DECODER_H_
#define INFERX_MODELS_LAYERS_DECODER_H_

#include <optional>
#include <variant>
#include <vector>

#include "inferx/config/parallel_config.h"
#include "inferx/core/status.h"
#include "inferx/core/tensor.h"
#include "inferx/dist/comm.h"
#include "inferx/models/layers/attention.h"
#include "inferx/models/layers/embedding.h"
#include "inferx/models/layers/gated_delta_net.h"
#include "inferx/models/layers/gated_mlp.h"
#include "inferx/models/layers/mla.h"
#include "inferx/models/layers/moe.h"
#include "inferx/models/layers/normalization.h"
#include "inferx/models/model.h"
#include "inferx/models/requirements.h"
#include "inferx/ops/op_context.h"

namespace inferx::models::layers {

/// \brief Where a layer's normalization sits relative to its residual adds.
enum class ResidualStyle {
  /// Llama lineage: normalize the residual stream before each sublayer, add
  /// the sublayer output raw (fused into the next AddForward).
  kPreNorm,
  /// Gemma lineage: normalize each sublayer output before it joins the
  /// residual stream (post_attention/pre_feedforward/post_feedforward
  /// norms).
  kOutputNorm,
};

/// \brief Configuration of one decoder layer: mixer and feed-forward
///        selection as data. Heterogeneous stacks cost a variant visit per
///        layer, not a new execution path.
struct DecoderLayerConfig {
  NormConfig norm;  ///< Applied before the mixer and before the feed-forward.
  /// Output-side norm on the mixer result; read only when residual is
  /// kOutputNorm.
  NormConfig mixer_out_norm;
  /// Output-side norm on the feed-forward result; same condition.
  NormConfig feed_forward_out_norm;
  ResidualStyle residual = ResidualStyle::kPreNorm;
  /// Token-mixer selection; index order matches DecoderLayerWeights::mixer.
  using Mixer = std::variant<AttentionConfig, MlaConfig, GatedDeltaNetConfig>;
  /// Feed-forward selection; index order matches
  /// DecoderLayerWeights::feed_forward.
  using FeedForward = std::variant<GatedMlpConfig, MoeConfig>;
  Mixer mixer;
  FeedForward feed_forward;
};

/// \brief Weights of one decoder layer.
struct DecoderLayerWeights {
  Tensor input_norm;       ///< [hidden] mixer-side norm.
  Tensor post_mixer_norm;  ///< [hidden] feed-forward-side norm.
  /// Output-side norm (Gemma sandwich); absent when residual is kPreNorm.
  std::optional<Tensor> mixer_out_norm;
  std::optional<Tensor> feed_forward_out_norm;  ///< Same condition.
  std::variant<AttentionWeights, MlaWeights, GdnWeights> mixer;
  std::variant<GatedMlpWeights, MoeWeights> feed_forward;
};

/// \brief Runs whichever feed-forward the layer configures into
///        `mixed_out`. The hot path stays a direct call.
Status RunFeedForward(const std::variant<GatedMlpConfig, MoeConfig>& config,
                      const std::variant<GatedMlpWeights, MoeWeights>& weights,
                      const Tensor& normed, GatedMlpWorkspace& mlp_ws, MoeWorkspace& moe_ws,
                      Tensor* packed_buffer, ops::OpContext& ctx, Tensor& mixed_out);

/// \brief Everything needed to build a decoder stack. Global dimensions
///        stay immutable here; ShardStackForRank derives the rank-local
///        copy separately.
struct StackConfig {
  int64_t hidden_size = 0;
  int64_t vocab_size = 0;  ///< Full vocabulary width across all ranks.
  NormConfig final_norm;   ///< Closing norm after the last layer.
  /// Activation multiplier applied after the embedding gather (Gemma:
  /// sqrt(hidden)).
  float embedding_scale = 1.0f;
  std::vector<DecoderLayerConfig> blocks;  ///< One entry per layer.

  /// \brief Checks internal consistency of dimensions, geometry, and
  ///        variants.
  Status Validate() const;

  /// \brief Checks that every configured variant has an executable
  ///        implementation -- run at build time so an unexecutable model is
  ///        rejected before any weight is loaded, never at first Forward.
  Status ValidateExecutable() const;

  /// Persistent state each layer needs, one entry per layer.
  std::vector<LayerStateSpec> StateRequirements() const;
};

/// \brief All stack weights, on the model's device.
struct StackWeights {
  Tensor token_embedding;  ///< This rank's shard of [vocab, hidden].
  Tensor final_norm;       ///< [hidden]
  std::vector<DecoderLayerWeights> blocks;
};

/// \brief Rewrites `config` rank-local: head counts, feed-forward widths,
///        and the vocabulary shard. Loading keeps the total config (this
///        rank's shard rows derive from it); the executed stack receives
///        the rewritten copy so workspaces, KV layouts, and kernels see
///        exactly this rank's dimensions. MLA and recurrent mixers are
///        rejected until their shard geometry is implemented.
Status ShardStackForRank(StackConfig& config, const ParallelConfig& parallel);

/// \brief Owns one layer's weights. The stack lends activation workspace
///        and the runner lends persistent state.
class DecoderLayer {
 public:
  DecoderLayer(const DecoderLayerConfig& config, DecoderLayerWeights weights);

  using LayerState = std::variant<PagedKvState, RecurrentState>;
  /// \param first  True for layer zero, whose input norm reads the raw
  ///        embedding (later layers fuse the residual add).
  Status Forward(bool first, Tensor& hidden, Tensor& normed, Tensor& mixed,
                 const AttentionBatch& batch, const LayerState& state,
                 const KvBlockPool* pool, const RecurrentStatePool* recurrent,
                 AttentionWorkspace& attention_ws,
                 MlaWorkspace& mla_ws, GdnWorkspace& gdn_ws, GatedMlpWorkspace& mlp_ws,
                 MoeWorkspace& moe_ws, Tensor& packed, ops::OpContext& ctx,
                 dist::CommBackend& comm) const;

 private:
  static std::optional<RmsNorm> MakeOutNorm(ResidualStyle residual, const NormConfig& config,
                                            std::optional<Tensor> weight);

  RmsNorm input_norm_;
  RmsNorm ffn_norm_;
  std::optional<RmsNorm> mixer_out_norm_;
  std::optional<RmsNorm> feed_forward_out_norm_;
  std::variant<AttentionConfig, MlaConfig, GatedDeltaNetConfig> mixer_;
  std::variant<GatedMlpConfig, MoeConfig> feed_forward_;
  std::variant<AttentionWeights, MlaWeights, GdnWeights> mixer_weights_;
  std::variant<GatedMlpWeights, MoeWeights> feed_forward_weights_;
  float norm_eps_;
  bool output_norm_residual_;
};

/// \brief The shared decoder stack: embedding, layers, final norm, and all
///        reusable activation workspace sized once for the configured
///        capacity.
class DecoderStack {
 public:
  DecoderStack(StackConfig config, StackWeights weights, int max_tokens);

  const StackConfig& config() const { return config_; }
  std::vector<LayerStateSpec> StateRequirements() const { return config_.StateRequirements(); }

  /// \brief Allocates every workspace for the execution limits and freezes
  ///        the workspace generation. Explicit counterpart to the lazy
  ///        first-forward allocation, which remains only as a fallback for
  ///        direct (non-runner) callers.
  Status Prepare(int max_tokens, int max_sequences, DeviceId device);

  /// \brief Weights and activation-workspace bytes held on device.
  int64_t weights_bytes() const;
  int64_t workspace_bytes() const;

  /// \brief Increments on every workspace (re)allocation; captured graphs
  ///        keyed on an older generation are invalid.
  uint64_t workspace_generation() const { return workspace_generation_; }

  bool prepared() const { return workspace_ready_; }
  int limits_tokens() const { return limits_tokens_; }
  int limits_sequences() const { return limits_sequences_; }
  int limits_logit_rows() const { return limits_sequences_; }
  const Tensor& embedding_weight() const { return embedding_.weight(); }

  /// \brief Embeds `token_ids`, runs every layer, applies the final norm.
  ///
  /// \return [num_tokens, hidden] final hidden states borrowing stack
  ///         workspace; valid until the next Forward.
  StatusOr<Tensor> Forward(const Tensor& token_ids, const AttentionBatch& batch,
                           ModelState& state, ops::OpContext& ctx, dist::CommBackend& comm);

 private:
  Status InitWorkspace(DeviceId device);
  StatusOr<Tensor> BeginForward(const Tensor& token_ids, const AttentionBatch& batch,
                                ModelState& state, ops::OpContext& ctx, dist::CommBackend& comm);

  StackConfig config_;
  TokenEmbedding embedding_;
  int max_tokens_;
  bool workspace_ready_ = false;
  uint64_t workspace_generation_ = 0;
  int64_t weights_bytes_ = 0;
  int64_t workspace_bytes_ = 0;
  int limits_tokens_ = 0;
  int limits_sequences_ = 0;
  int64_t max_intermediate_ = 0;
  int64_t max_mla_heads_ = 0;
  int64_t max_mla_head_dim_ = 0;
  int64_t max_mla_q_lora_ = 0;
  int64_t max_mla_kv_lora_ = 0;
  int64_t max_mla_rope_ = 0;
  int64_t max_gdn_proj_ = 0;
  int64_t max_gdn_conv_ = 0;
  int64_t max_gdn_value_ = 0;
  int64_t max_gdn_heads_ = 0;
  int64_t max_mla_up_width_ = 0;
  int64_t max_moe_experts_ = 0;
  int64_t max_experts_per_token_ = 0;
  int64_t max_moe_intermediate_ = 0;
  int64_t max_shared_intermediate_ = 0;
  bool enable_split_decode_ = false;
  int prefill_tile_rows_ = 64;
  std::optional<Tensor> packed_projection_;
  std::optional<Tensor> hidden_, normed_, mixed_;
  // Always constructed; buffers inside stay empty until a layer of that
  // kind exists (the workspace scan decides), and unused workspaces are
  // never touched by any layer.
  AttentionWorkspace attention_;
  MlaWorkspace mla_;
  GdnWorkspace gdn_;
  GatedMlpWorkspace mlp_;
  MoeWorkspace moe_;
  bool has_mla_ = false;
  bool has_gdn_ = false;
  bool has_moe_ = false;
  std::vector<DecoderLayer> layers_;
  RmsNorm final_norm_;
};

}  // namespace inferx::models::layers

#endif  // INFERX_MODELS_LAYERS_DECODER_H_
