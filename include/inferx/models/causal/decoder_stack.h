#pragma once

#include <optional>
#include <utility>
#include <vector>

#include "inferx/models/causal/decoder_config.h"
#include "inferx/models/causal/decoder_layer.h"

namespace inferx::causal {

/// Shared allocation, input validation, and attention planning. Keeping this
/// outside the template avoids duplicating workspace machinery per family.
class DecoderWorkspace {
 protected:
  DecoderWorkspace(DecoderConfig config, Tensor embedding, int max_tokens);
  Status InitWorkspace(DeviceId device);
  StatusOr<Tensor> BeginForward(const DecoderInput& input, ModelState& state,
                                ops::OpContext& ctx, dist::CommBackend& comm);

  DecoderConfig config_;
  Tensor embedding_;
  int max_tokens_;
  bool workspace_ready_ = false;
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
  std::optional<components::AttentionWorkspace> attention_;
  std::optional<components::MlaWorkspace> mla_;
  std::optional<components::GdnWorkspace> gdn_;
  std::optional<components::MlpWorkspace> mlp_;
  std::optional<components::MoeWorkspace> moe_;
};

template <ModelTraits Traits>
class DecoderStack final : private DecoderWorkspace {
 public:
  DecoderStack(DecoderConfig config, DecoderWeights weights, int max_tokens)
      : DecoderWorkspace(std::move(config), std::move(weights.token_embedding), max_tokens),
        final_norm_(config_.final_norm, std::move(weights.final_norm)) {
    layers_.reserve(config_.blocks.size());
    for (size_t i = 0; i < config_.blocks.size(); ++i) {
      layers_.emplace_back(config_.blocks[i], std::move(weights.blocks[i]));
    }
  }

  const CheckpointConfig& config() const { return config_.model; }
  std::vector<LayerStateSpec> StateRequirements() const { return config_.StateRequirements(); }

  StatusOr<Tensor> Forward(const DecoderInput& input, ModelState& state,
                           ops::OpContext& ctx, dist::CommBackend& comm) {
    INFERX_ASSIGN_OR_RETURN(Tensor hidden, BeginForward(input, state, ctx, comm));
    INFERX_ASSIGN_OR_RETURN(Tensor normed, normed_->Slice(0, input.attention.num_tokens));
    INFERX_ASSIGN_OR_RETURN(Tensor mixed, mixed_->Slice(0, input.attention.num_tokens));
    for (size_t i = 0; i < layers_.size(); ++i) {
      INFERX_RETURN_IF_ERROR(layers_[i].Forward(
          i == 0, hidden, normed, mixed, input.attention, state.layers[i], *state.paged_kv,
          state.recurrent == nullptr ? *gdn_fallback_pool_ : *state.recurrent, *attention_,
          *mla_, *gdn_, *mlp_, *moe_, *packed_projection_, ctx, comm));
    }
    INFERX_RETURN_IF_ERROR(final_norm_.AddForward(ctx, mixed, hidden, normed));
    return normed;
  }

 private:
  // Present only for models with no recurrent layers, so the layer's pool
  // reference is never dangling; recurrent models use the runner's pool.
  static constexpr RecurrentStatePool* kNoRecurrentPool = nullptr;
  RecurrentStatePool* gdn_fallback_pool_ = const_cast<RecurrentStatePool*>(
      static_cast<const RecurrentStatePool*>(kNoRecurrentPool));
  std::vector<DecoderLayer<Traits>> layers_;
  typename Traits::Norm final_norm_;
};

}  // namespace inferx::causal
