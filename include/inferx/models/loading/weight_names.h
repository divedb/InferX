#pragma once

#include <string_view>

namespace inferx::models {

/// Module paths without .weight/.bias. Layer-local paths are relative to
/// `layers.<index>.`; embedding, final norm, and head paths are absolute.
struct WeightNames {
  std::string_view embed = "model.embed_tokens";
  std::string_view layers = "model.layers";
  std::string_view final_norm = "model.norm";
  std::string_view head = "lm_head";
  std::string_view attn_norm = "input_layernorm";
  std::string_view ffn_norm = "post_attention_layernorm";
  std::string_view q = "self_attn.q_proj";
  std::string_view k = "self_attn.k_proj";
  std::string_view v = "self_attn.v_proj";
  std::string_view qkv = "";
  std::string_view o = "self_attn.o_proj";
  std::string_view q_norm = "self_attn.q_norm";
  std::string_view k_norm = "self_attn.k_norm";
  std::string_view gate = "mlp.gate_proj";
  std::string_view up = "mlp.up_proj";
  std::string_view down = "mlp.down_proj";
  std::string_view experts = "mlp.experts";
  std::string_view router = "mlp.gate";
  std::string_view shared_expert = "mlp.shared_expert";
  std::string_view shared_expert_gate = "mlp.shared_expert_gate";
};

/// Source storage only. Execution always uses packed, block-contiguous Q/K/V.
enum class QkvLayout { kSeparate, kFused };

struct WeightLayout {
  QkvLayout qkv = QkvLayout::kSeparate;
  // Both supported layouts use [out, in] matrices; fused rows are [Q | K | V].
};

}  // namespace inferx::models
