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
  std::string_view sinks = "self_attn.sinks";
  std::string_view k_norm = "self_attn.k_norm";
  std::string_view gate = "mlp.gate_proj";
  std::string_view up = "mlp.up_proj";
  std::string_view down = "mlp.down_proj";
  std::string_view experts = "mlp.experts";
  std::string_view router = "mlp.gate";
  std::string_view shared_expert = "mlp.shared_expert";
  std::string_view shared_expert_gate = "mlp.shared_expert_gate";
  /// DeepSeek's selection-only router bias.
  std::string_view correction_bias = "mlp.e_score_correction_bias";
  /// gpt-oss stores every expert fused and MXFP4-quantized: packed nibbles,
  /// E8M0 block scales, and interleaved gate/up rows (and biases).
  std::string_view experts_gate_up_blocks = "mlp.experts.gate_up_proj_blocks";
  std::string_view experts_gate_up_scales = "mlp.experts.gate_up_proj_scales";
  std::string_view experts_gate_up_bias = "mlp.experts.gate_up_proj_bias";
  std::string_view experts_down_blocks = "mlp.experts.down_proj_blocks";
  std::string_view experts_down_scales = "mlp.experts.down_proj_scales";
  std::string_view experts_down_bias = "mlp.experts.down_proj_bias";
  /// Output-side norms (Gemma's sandwich); empty disables.
  std::string_view attn_out_norm = "";
  std::string_view ffn_out_norm = "";
  /// Multi-latent attention projections (DeepSeek).
  std::string_view mla_q_a = "self_attn.q_a_proj";
  std::string_view mla_q_a_norm = "self_attn.q_a_layernorm";
  std::string_view mla_q_b = "self_attn.q_b_proj";
  std::string_view mla_kv_a = "self_attn.kv_a_proj_with_mqa";
  std::string_view mla_kv_a_norm = "self_attn.kv_a_layernorm";
  std::string_view mla_kv_b = "self_attn.kv_b_proj";
  /// Gated DeltaNet projections (Qwen3-Next).
  std::string_view gdn_qkvz = "self_attn.in_proj_qkvz";
  std::string_view gdn_ba = "self_attn.in_proj_ba";
  std::string_view gdn_conv = "self_attn.conv1d";
  std::string_view gdn_a_log = "self_attn.A_log";
  std::string_view gdn_dt_bias = "self_attn.dt_bias";
  std::string_view gdn_norm = "self_attn.norm";
  std::string_view gdn_out = "self_attn.out_proj";
};

/// Source storage only. Execution always uses packed, block-contiguous Q/K/V.
enum class QkvLayout { kSeparate, kFused };

struct WeightLayout {
  QkvLayout qkv = QkvLayout::kSeparate;
  // Both supported layouts use [out, in] matrices; fused rows are [Q | K | V].
};

}  // namespace inferx::models
