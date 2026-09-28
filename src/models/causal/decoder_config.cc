#include "inferx/models/causal/decoder_config.h"

#include <cmath>

#include "inferx/ops/attention.h"

namespace inferx::causal {

Status DecoderConfig::Validate() const {
  if (model.hidden_size <= 0 || model.vocab_size <= 0 || model.num_hidden_layers <= 0 ||
      blocks.size() != static_cast<size_t>(model.num_hidden_layers)) {
    return InvalidArgumentError("decoder dimensions and block count disagree");
  }
  if (!std::isfinite(final_norm.eps) || final_norm.eps < 0) {
    return InvalidArgumentError("invalid final normalization epsilon");
  }
  for (const auto& block : blocks) {
    if (!std::isfinite(block.norm.eps) || block.norm.eps < 0) {
      return InvalidArgumentError("invalid block normalization epsilon");
    }
    if (const auto* m = std::get_if<components::MlaConfig>(&block.mixer)) {
      if (m->query_heads <= 0 || m->q_lora_rank <= 0 || m->kv_lora_rank <= 0 ||
          m->qk_nope_head_dim <= 0 || m->qk_rope_head_dim <= 0 || m->v_head_dim <= 0 ||
          m->v_head_dim > m->head_dim() || m->rotary.dim != m->qk_rope_head_dim) {
        return InvalidArgumentError("invalid multi-latent attention geometry");
      }
    } else if (const auto* a = std::get_if<components::AttentionConfig>(&block.mixer)) {
      if (a->query_heads <= 0 || a->kv_heads <= 0 || a->head_dim <= 0 ||
          a->query_heads % a->kv_heads != 0 || a->rotary.dim <= 0 ||
          a->rotary.dim > a->head_dim || a->rotary.dim % 2 != 0 ||
          !std::isfinite(a->rotary.theta) || a->rotary.theta <= 0 ||
          !std::isfinite(a->rotary.factor) || a->rotary.factor <= 0 || a->sliding_window < 0) {
        return InvalidArgumentError("invalid attention geometry or rotary configuration");
      }
    } else {
      const auto& g = std::get<components::GatedDeltaNetConfig>(block.mixer);
      if (g.key_heads <= 0 || g.value_heads <= 0 || g.key_dim <= 0 || g.value_dim <= 0 ||
          g.value_heads % g.key_heads != 0 || g.conv_kernel_size <= 0) {
        return InvalidArgumentError("invalid Gated DeltaNet geometry");
      }
    }
    if (const auto* dense = std::get_if<components::SwiGluConfig>(&block.feed_forward)) {
      if (dense->intermediate_size <= 0) return InvalidArgumentError("invalid SwiGLU width");
    } else {
      const auto& moe = std::get<components::MoeConfig>(block.feed_forward);
      if (moe.num_experts <= 0 || moe.experts_per_token <= 0 ||
          moe.experts_per_token > moe.num_experts || moe.intermediate_size <= 0 ||
          moe.shared_intermediate_size < 0 ||
          (moe.gate_shared_expert && moe.shared_intermediate_size == 0)) {
        return InvalidArgumentError("invalid expert routing configuration");
      }
    }
  }
  return OkStatus();
}

Status DecoderConfig::ValidateExecutable() const {
  for (const auto& block : blocks) {
    const auto* a = std::get_if<components::AttentionConfig>(&block.mixer);
    if (a == nullptr && !std::holds_alternative<components::MlaConfig>(block.mixer) &&
        !std::holds_alternative<components::GatedDeltaNetConfig>(block.mixer)) {
      return UnimplementedError("mixer execution is not implemented");
    }
    if (a != nullptr) {
      ops::AttentionParams params;
      params.query_heads = a->query_heads;
      params.kv_heads = a->kv_heads;
      params.head_dim = a->head_dim;
      params.scale = a->scale_override > 0.0f ? a->scale_override : 1.0f;
      params.sliding_window = a->sliding_window;
      INFERX_RETURN_IF_ERROR(ops::ValidateAttentionGeometry(params));
      if (a->head_dim > 8 * 256) {
        return UnimplementedError("attention head dimension exceeds the generic kernel");
      }
    }
    if (const auto* moe = std::get_if<components::MoeConfig>(&block.feed_forward)) {
      if (moe->num_experts > 512 || moe->experts_per_token > 64) {
        return UnimplementedError("expert counts exceed the routing kernel");
      }
      if (moe->routing.group_count > 1) {
        const int64_t per_group = moe->num_experts / moe->routing.group_count;
        if (moe->num_experts % moe->routing.group_count != 0 ||
            moe->routing.group_topk * per_group < moe->experts_per_token) {
          return InvalidArgumentError("grouped routing cannot select enough experts");
        }
      }
    }
  }
  return OkStatus();
}

std::vector<LayerStateSpec> DecoderConfig::StateRequirements() const {
  std::vector<LayerStateSpec> specs;
  specs.reserve(blocks.size());
  for (const auto& block : blocks) {
    if (const auto* a = std::get_if<components::AttentionConfig>(&block.mixer)) {
      KvLayout layout;
      layout.kv_heads = a->kv_heads;
      layout.head_dim = a->head_dim;
      layout.dtype = DataType::kBFloat16;
      specs.push_back(PagedKvStateSpec{layout});
    } else if (const auto* m = std::get_if<components::MlaConfig>(&block.mixer)) {
      // Decompressed MLA caches per-head K and zero-padded V at head_dim.
      specs.push_back(PagedKvStateSpec{KvLayout{2, m->query_heads, m->head_dim(),
                                                 DataType::kBFloat16}});
    } else {
      const auto& g = std::get<components::GatedDeltaNetConfig>(block.mixer);
      specs.push_back(RecurrentStateSpec{g.key_heads, g.value_heads, g.key_dim, g.value_dim,
                                         g.conv_kernel_size});
    }
  }
  return specs;
}

}  // namespace inferx::causal
