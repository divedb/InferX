#include "inferx/models/lm/stack.h"
#include "models/lm/diagnostic_trace.h"

#include <cmath>
#include <cstdlib>
#include <string_view>
#include <utility>
#include <vector>

#include "inferx/core/device_runtime.h"
#include "inferx/core/shape.h"
#include "inferx/models/state.h"
#include "inferx/ops/attention.h"
#include "inferx/ops/flash_attention.h"
#include "inferx/ops/elementwise.h"
#include "inferx/ops/gather.h"
#include "inferx/ops/linear.h"
#include "inferx/ops/rotary.h"
#include "inferx/ops/rms_norm.h"

namespace inferx::lm {

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
    if (const auto* a = std::get_if<layers::AttentionConfig>(&block.mixer)) {
      if (a->query_heads <= 0 || a->kv_heads <= 0 || a->head_dim <= 0 ||
          a->query_heads % a->kv_heads != 0 || a->rotary.dim <= 0 ||
          a->rotary.dim > a->head_dim || a->rotary.dim % 2 != 0 ||
          !std::isfinite(a->rotary.theta) || a->rotary.theta <= 0 ||
          !std::isfinite(a->rotary.factor) || a->rotary.factor <= 0 ||
          a->sliding_window < 0) {
        return InvalidArgumentError("invalid attention geometry or rotary configuration");
      }
    } else {
      const auto& g = std::get<layers::GatedDeltaNetConfig>(block.mixer);
      if (g.key_heads <= 0 || g.value_heads <= 0 || g.key_dim <= 0 || g.value_dim <= 0 ||
          g.value_heads % g.key_heads != 0 || g.conv_kernel_size <= 0) {
        return InvalidArgumentError("invalid Gated DeltaNet geometry");
      }
    }
    if (const auto* dense = std::get_if<layers::SwiGluConfig>(&block.feed_forward)) {
      if (dense->intermediate_size <= 0) return InvalidArgumentError("invalid SwiGLU width");
    } else {
      const auto& moe = std::get<layers::MoeConfig>(block.feed_forward);
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

std::vector<LayerStateSpec> DecoderConfig::StateRequirements() const {
  std::vector<LayerStateSpec> specs;
  specs.reserve(blocks.size());
  for (const auto& block : blocks) {
    if (const auto* a = std::get_if<layers::AttentionConfig>(&block.mixer)) {
      KvLayout layout;
      layout.kv_heads = a->kv_heads;
      layout.head_dim = a->head_dim;
      layout.dtype = DataType::kBFloat16;
      specs.push_back(PagedKvStateSpec{layout});
    } else {
      const auto& g = std::get<layers::GatedDeltaNetConfig>(block.mixer);
      specs.push_back(RecurrentStateSpec{g.key_heads, g.value_heads, g.key_dim,
                                         g.value_dim, g.conv_kernel_size});
    }
  }
  return specs;
}

DecoderStack::DecoderStack(DecoderConfig config, DecoderWeights weights, int max_tokens)
    : config_(std::move(config)), weights_(std::move(weights)), max_tokens_(max_tokens) {
  // Experimental until full-model numerical parity is established. Read once
  // per model so graph capture and replay always use the same implementation.
  const char* flag = std::getenv("INFERX_EXPERIMENTAL_SPLIT_DECODE");
  enable_split_decode_ = flag != nullptr && std::string_view(flag) == "1";
  flag = std::getenv("INFERX_EXPERIMENTAL_PREFILL_TILE");
  if (flag != nullptr && std::string_view(flag) == "128") prefill_tile_rows_ = 128;
}

Status DecoderStack::InitWorkspace(DeviceId device) {
  const auto& dims = config_.model;
  int64_t query_dim = 0;
  int64_t kv_dim = 0;
  int64_t query_heads = 0;
  for (const auto& block : config_.blocks) {
    if (const auto* a = std::get_if<layers::AttentionConfig>(&block.mixer)) {
      query_heads = std::max(query_heads, a->query_heads);
      query_dim = std::max<int64_t>(query_dim, a->query_heads * a->head_dim);
      kv_dim = std::max<int64_t>(kv_dim, a->kv_heads * a->head_dim);
    }
    if (const auto* dense = std::get_if<layers::SwiGluConfig>(&block.feed_forward)) {
      max_intermediate_ = std::max(max_intermediate_, dense->intermediate_size);
    }
  }
  const int64_t rows = max_tokens_;
  const auto alloc2 = [&](int64_t cols) {
    return Tensor::Empty(DataType::kBFloat16, Shape({rows, cols}), device);
  };
  const auto alloc_flat = [&](int64_t cols) {
    return Tensor::Empty(DataType::kBFloat16, Shape({rows * cols}), device);
  };
  INFERX_ASSIGN_OR_RETURN(hidden_, alloc2(dims.hidden_size));
  INFERX_ASSIGN_OR_RETURN(normed_, alloc2(dims.hidden_size));
  INFERX_ASSIGN_OR_RETURN(query_, alloc_flat(query_dim));
  INFERX_ASSIGN_OR_RETURN(key_, alloc_flat(kv_dim));
  INFERX_ASSIGN_OR_RETURN(value_, alloc_flat(kv_dim));
  INFERX_ASSIGN_OR_RETURN(attn_out_, alloc_flat(query_dim));
  INFERX_ASSIGN_OR_RETURN(mixed_, alloc2(dims.hidden_size));
  INFERX_ASSIGN_OR_RETURN(gate_, alloc_flat(max_intermediate_));
  INFERX_ASSIGN_OR_RETURN(up_, alloc_flat(max_intermediate_));
  bool packed = false;
  for (const auto& weights : weights_.blocks) {
    packed = packed || weights.mixer.packed_qkv.IsDefined();
    if (const auto* ffn = std::get_if<layers::SwiGluWeights>(&weights.feed_forward))
      packed = packed || ffn->packed_gate_up.IsDefined();
  }
  if (packed) {
    INFERX_ASSIGN_OR_RETURN(packed_projection_,
        alloc_flat(std::max(query_dim + 2 * kv_dim, 2 * max_intermediate_)));
  }
  INFERX_ASSIGN_OR_RETURN(attention_plan_,
      Tensor::Empty(DataType::kInt32, Shape({3 * rows + 1}), device));
  if (enable_split_decode_) {
    constexpr int batch = ops::FlashDecodeWorkspace::kMaxBatch;
    constexpr int tiles = batch * ops::FlashDecodeWorkspace::kPartitions;
    INFERX_ASSIGN_OR_RETURN(decode_workspace_.plan,
        Tensor::Empty(DataType::kInt32, Shape({3 * tiles + batch + 2}), device));
    INFERX_ASSIGN_OR_RETURN(decode_workspace_.values,
        Tensor::Empty(DataType::kBFloat16, Shape({tiles * query_dim}), device));
    INFERX_ASSIGN_OR_RETURN(decode_workspace_.scores,
        Tensor::Empty(DataType::kFloat, Shape({tiles * query_heads}), device));
  }
  workspace_ready_ = true;
  return OkStatus();
}

StatusOr<Tensor> DecoderStack::Forward(const DecoderInput& input, ModelState& state,
                                       ops::ExecutionContext& ctx) {
  const int rows = input.attention.num_tokens;
  DiagnosticTrace trace(ctx);
  if (rows <= 0 || rows > max_tokens_) {
    return InvalidArgumentError("decoder token count exceeds workspace capacity");
  }
  if (input.embeddings.IsDefined()) {
    if (input.embeddings.Rank() != 2 || input.embeddings.Dim(0) != rows ||
        input.embeddings.Dim(1) != config_.model.hidden_size ||
        input.embeddings.Device() != ctx.device()) {
      return InvalidArgumentError("invalid prepared decoder embeddings");
    }
  } else if (!input.token_ids.IsDefined() || input.token_ids.Rank() != 1 ||
             input.token_ids.Numel() != rows || input.token_ids.GetDataType() != DataType::kInt32 ||
             input.token_ids.Device() != ctx.device()) {
    return InvalidArgumentError("invalid decoder token input");
  }
  if (state.layers.size() != config_.blocks.size()) {
    return InvalidArgumentError("decoder state must contain one entry per layer");
  }
  const auto specs = config_.StateRequirements();
  for (size_t i = 0; i < specs.size(); ++i) {
    if (std::holds_alternative<PagedKvStateSpec>(specs[i])) {
      const auto* entry = std::get_if<PagedKvState>(&state.layers[i]);
      if (entry == nullptr || state.paged_kv == nullptr || entry->pool_layer < 0 ||
          entry->pool_layer >= state.paged_kv->num_layers()) {
        return InvalidArgumentError("missing paged state for layer ", i);
      }
    } else if (!std::holds_alternative<RecurrentState>(state.layers[i])) {
      return InvalidArgumentError("missing recurrent state for layer ", i);
    }
  }
  if (!workspace_ready_) {
    INFERX_RETURN_IF_ERROR(InitWorkspace(ctx.device()));
  }

  // Embed tokens, or accept prepared embeddings, into the hidden workspace.
  INFERX_ASSIGN_OR_RETURN(Tensor hidden, hidden_.Slice(0, rows));
  if (input.embeddings.IsDefined()) {
    INFERX_RETURN_IF_ERROR(ctx.runtime().CopyAsync(
        hidden.Data(), input.embeddings.Data(), hidden.NBytes(), CopyKind::kDeviceToDevice,
        ctx.stream()));
  } else {
    INFERX_RETURN_IF_ERROR(ops::GatherRows(ctx, weights_.token_embedding, input.token_ids, hidden));
  }

  const auto& attention_batch = input.attention;
  if (trace.enabled()) trace.Write("embedding", hidden);
  if (attention_batch.num_seqs <= 0 ||
      attention_batch.host_qo_indptr.size() != static_cast<size_t>(attention_batch.num_seqs + 1) ||
      attention_batch.host_qo_indptr.front() != 0 || attention_batch.host_qo_indptr.back() != rows) {
    return InvalidArgumentError("attention requires host query offsets matching the batch");
  }
  for (int seq = 0; seq < attention_batch.num_seqs; ++seq) {
    if (attention_batch.host_qo_indptr[seq + 1] <= attention_batch.host_qo_indptr[seq])
      return InvalidArgumentError("attention sequences must have positive query lengths");
  }
  int planned_group = 0;
  int attention_tiles = 0;
  const bool split_decode = enable_split_decode_ && rows == attention_batch.num_seqs &&
      rows <= ops::FlashDecodeWorkspace::kMaxBatch;
  if (split_decode) {
    INFERX_RETURN_IF_ERROR(ops::PrepareFlashDecode(ctx, attention_batch.kv_indptr,
        attention_batch.last_page_len, state.paged_kv->block_size(), decode_workspace_));
  }
  for (size_t i = 0; i < config_.blocks.size(); ++i) {
    const std::string prefix = trace.enabled() ? "layer_" + std::to_string(i) + "." : "";
    const auto& block = config_.blocks[i];
    const auto& weights = weights_.blocks[i];
    if (!std::holds_alternative<layers::AttentionConfig>(block.mixer)) {
      return UnimplementedError("recurrent mixer execution is not implemented");
    }
    const layers::AttentionConfig& a = std::get<layers::AttentionConfig>(block.mixer);
    if (a.output_gate != layers::OutputGate::kNone) {
      return UnimplementedError("gated attention output is not implemented");
    }
    if (a.projection_bias) {
      return UnimplementedError("biased projections are not implemented");
    }
    INFERX_ASSIGN_OR_RETURN(Tensor normed, normed_.Slice(0, rows));
    ops::RMSNormConfig norm{block.norm.eps, block.norm.plus_one, !block.norm.plus_one};
    if (i == 0) {
      INFERX_RETURN_IF_ERROR(ops::RmsNorm(ctx, hidden, weights.input_norm, normed, norm));
    } else {
      INFERX_ASSIGN_OR_RETURN(Tensor previous_mixed, mixed_.Slice(0, rows));
      INFERX_RETURN_IF_ERROR(ops::AddRmsNorm(ctx, previous_mixed, hidden,
                                            weights.input_norm, normed, norm));
    }

    // Attention mixer: project, normalize q/k per head, rotate, cache, attend.
    if (trace.enabled()) trace.Write(prefix + "input_norm", normed);
    const int64_t query_width = a.query_heads * a.head_dim;
    const int64_t kv_width = a.kv_heads * a.head_dim;
    INFERX_ASSIGN_OR_RETURN(Tensor q_flat, query_.Slice(0, rows * query_width));
    INFERX_ASSIGN_OR_RETURN(Tensor q, q_flat.Reshape(Shape({rows, query_width})));
    INFERX_ASSIGN_OR_RETURN(Tensor k_flat, key_.Slice(0, rows * kv_width));
    INFERX_ASSIGN_OR_RETURN(Tensor k, k_flat.Reshape(Shape({rows, kv_width})));
    INFERX_ASSIGN_OR_RETURN(Tensor v_flat, value_.Slice(0, rows * kv_width));
    INFERX_ASSIGN_OR_RETURN(Tensor v, v_flat.Reshape(Shape({rows, kv_width})));
    if (weights.mixer.packed_qkv.IsDefined()) {
      const int64_t width = query_width + 2 * kv_width;
      INFERX_ASSIGN_OR_RETURN(auto flat, packed_projection_.Slice(0, rows * width));
      INFERX_ASSIGN_OR_RETURN(auto packed, flat.Reshape(Shape({rows, width})));
      INFERX_RETURN_IF_ERROR(ops::Linear(ctx, normed, weights.mixer.packed_qkv, packed));
      INFERX_RETURN_IF_ERROR(ops::SplitQkv(ctx, packed, q, k, v));
    } else {
      INFERX_RETURN_IF_ERROR(ops::Linear(ctx, normed, weights.mixer.query.weight, q));
      INFERX_RETURN_IF_ERROR(ops::Linear(ctx, normed, weights.mixer.key.weight, k));
      INFERX_RETURN_IF_ERROR(ops::Linear(ctx, normed, weights.mixer.value.weight, v));
    }
    if (trace.enabled()) trace.Write(prefix + "q_proj", q);
    if (trace.enabled()) trace.Write(prefix + "k_proj", k);
    if (trace.enabled()) trace.Write(prefix + "v_proj", v);
    const bool fused_norm_rope = a.qk_norm && a.head_dim == 128 && ctx.device().IsCuda();
    if (a.qk_norm && !fused_norm_rope) {
      INFERX_ASSIGN_OR_RETURN(Tensor q_heads,
                              q.Reshape(Shape({rows * a.query_heads, a.head_dim})));
      INFERX_RETURN_IF_ERROR(ops::RmsNorm(ctx, q_heads, weights.mixer.query_norm, q_heads,
                                          ops::RMSNormConfig{block.norm.eps, false, true}));
      INFERX_ASSIGN_OR_RETURN(Tensor k_heads,
                              k.Reshape(Shape({rows * a.kv_heads, a.head_dim})));
      INFERX_RETURN_IF_ERROR(ops::RmsNorm(ctx, k_heads, weights.mixer.key_norm, k_heads,
                                          ops::RMSNormConfig{block.norm.eps, false, true}));
    }
    INFERX_ASSIGN_OR_RETURN(Tensor q3, q.Reshape(Shape({rows, a.query_heads, a.head_dim})));
    INFERX_ASSIGN_OR_RETURN(Tensor k3, k.Reshape(Shape({rows, a.kv_heads, a.head_dim})));
    if (fused_norm_rope) {
      INFERX_RETURN_IF_ERROR(ops::NormalizeAndApplyRope(ctx, q3, k3,
          weights.mixer.query_norm, weights.mixer.key_norm, attention_batch.positions,
          block.norm.eps, ops::RotaryParams{a.rotary.dim, a.rotary.theta}));
    } else {
      INFERX_RETURN_IF_ERROR(
        ops::ApplyRope(ctx, q3, k3, attention_batch.positions,
                       ops::RotaryParams{a.rotary.dim, a.rotary.theta}));
    }

    const auto& layer_state = std::get<PagedKvState>(state.layers[i]);
    if (trace.enabled()) trace.Write(prefix + "q_rope", q);
    if (trace.enabled()) trace.Write(prefix + "k_rope", k);
    INFERX_ASSIGN_OR_RETURN(Tensor key_cache, state.paged_kv->KeyCache(layer_state.pool_layer));
    INFERX_ASSIGN_OR_RETURN(Tensor value_cache,
                            state.paged_kv->ValueCache(layer_state.pool_layer));
    const int64_t block_size = state.paged_kv->block_size();
    INFERX_RETURN_IF_ERROR(ops::WritePagedKv(ctx, k, v, attention_batch.positions,
                                             attention_batch.batch_indices,
                                             attention_batch.kv_indptr, attention_batch.kv_indices,
                                             key_cache, value_cache, block_size));
    INFERX_ASSIGN_OR_RETURN(Tensor attn_flat, attn_out_.Slice(0, rows * query_width));
    INFERX_ASSIGN_OR_RETURN(Tensor attn_out, attn_flat.Reshape(Shape({rows, query_width})));
    ops::AttentionParams params;
    params.query_heads = a.query_heads;
    params.kv_heads = a.kv_heads;
    params.head_dim = a.head_dim;
    params.scale = 1.0f / std::sqrt(static_cast<float>(a.head_dim));
    params.sliding_window = a.sliding_window;
    INFERX_RETURN_IF_ERROR(ops::ValidateAttentionGeometry(config_.attention_backend, params));
    const int group = a.query_heads / a.kv_heads;
    if (planned_group != group) {
      attention_tiles = 0;
      for (int seq = 0; seq < attention_batch.num_seqs; ++seq) {
        const int length = attention_batch.host_qo_indptr[seq + 1] - attention_batch.host_qo_indptr[seq];
        attention_tiles += (group * length + prefill_tile_rows_ - 1) / prefill_tile_rows_;
      }
      INFERX_RETURN_IF_ERROR(ops::PrepareFlashAttention(ctx, attention_batch.qo_indptr,
          attention_plan_, group, attention_tiles, prefill_tile_rows_));
      planned_group = group;
    }
    INFERX_RETURN_IF_ERROR(ops::FlashPagedAttention(ctx, q, attention_batch.qo_indptr,
        attention_batch.kv_indptr, attention_batch.kv_indices, attention_batch.last_page_len,
        key_cache, value_cache, block_size, params, attention_plan_, attention_tiles, attn_out,
        split_decode ? &decode_workspace_ : nullptr, prefill_tile_rows_));
    INFERX_ASSIGN_OR_RETURN(Tensor mixed, mixed_.Slice(0, rows));
    if (trace.enabled()) trace.Write(prefix + "attention", attn_out);
    INFERX_RETURN_IF_ERROR(ops::Linear(ctx, attn_out, weights.mixer.output.weight, mixed));
    if (trace.enabled()) trace.Write(prefix + "o_proj", mixed);
    // Feed-forward: norm, SwiGLU, project back, residual.
    INFERX_RETURN_IF_ERROR(ops::AddRmsNorm(ctx, mixed, hidden, weights.post_mixer_norm, normed, norm));
    if (trace.enabled()) trace.Write(prefix + "post_norm", normed);
    if (trace.enabled()) trace.Write(prefix + "residual", hidden);
    if (!std::holds_alternative<layers::SwiGluConfig>(block.feed_forward)) {
      return UnimplementedError("expert feed-forward execution is not implemented");
    }
    const auto& ffn = std::get<layers::SwiGluConfig>(block.feed_forward);
    const auto& ffn_weights = std::get<layers::SwiGluWeights>(weights.feed_forward);
    INFERX_ASSIGN_OR_RETURN(Tensor gate_flat, gate_.Slice(0, rows * ffn.intermediate_size));
    INFERX_ASSIGN_OR_RETURN(Tensor gate, gate_flat.Reshape(Shape({rows, ffn.intermediate_size})));
    INFERX_ASSIGN_OR_RETURN(Tensor up_flat, up_.Slice(0, rows * ffn.intermediate_size));
    INFERX_ASSIGN_OR_RETURN(Tensor up, up_flat.Reshape(Shape({rows, ffn.intermediate_size})));
    if (ffn_weights.packed_gate_up.IsDefined()) {
      const int64_t width = 2 * ffn.intermediate_size;
      INFERX_ASSIGN_OR_RETURN(auto flat, packed_projection_.Slice(0, rows * width));
      INFERX_ASSIGN_OR_RETURN(auto packed, flat.Reshape(Shape({rows, width})));
      INFERX_RETURN_IF_ERROR(ops::Linear(ctx, normed, ffn_weights.packed_gate_up, packed));
      INFERX_RETURN_IF_ERROR(ops::PackedSiluAndMul(ctx, packed, gate));
    } else {
      INFERX_RETURN_IF_ERROR(ops::Linear(ctx, normed, ffn_weights.gate.weight, gate));
      INFERX_RETURN_IF_ERROR(ops::Linear(ctx, normed, ffn_weights.up.weight, up));
      INFERX_RETURN_IF_ERROR(ops::SiluAndMul(ctx, gate, up, gate));
    }
    INFERX_RETURN_IF_ERROR(ops::Linear(ctx, gate, ffn_weights.down.weight, mixed));
    if (trace.enabled()) trace.Write(prefix + "silu", gate);
    if (trace.enabled()) trace.Write(prefix + "down_proj", mixed);
    // The residual addition is fused into the next block's normalization.
  }

  INFERX_ASSIGN_OR_RETURN(Tensor final_rows, normed_.Slice(0, rows));
  INFERX_ASSIGN_OR_RETURN(Tensor last_mixed, mixed_.Slice(0, rows));
  INFERX_RETURN_IF_ERROR(ops::AddRmsNorm(
      ctx, last_mixed, hidden, weights_.final_norm, final_rows,
      ops::RMSNormConfig{config_.final_norm.eps, config_.final_norm.plus_one,
                        !config_.final_norm.plus_one}));
  if (trace.enabled()) trace.Write("final_norm", final_rows);
  return final_rows;
}

}  // namespace inferx::lm
