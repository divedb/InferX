#include "inferx/models/causal/decoder_stack.h"
#include "models/diagnostic_trace.h"

#include <cmath>
#include <cstdlib>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "inferx/core/device_runtime.h"
#include "inferx/core/shape.h"
#include "inferx/models/state.h"
#include "inferx/ops/attention.h"
#include "inferx/ops/flash_attention.h"
#include "inferx/ops/gather.h"
#include "inferx/ops/rms_norm.h"

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
    if (const auto* a = std::get_if<components::AttentionConfig>(&block.mixer)) {
      if (a->query_heads <= 0 || a->kv_heads <= 0 || a->head_dim <= 0 ||
          a->query_heads % a->kv_heads != 0 || a->rotary.dim <= 0 ||
          a->rotary.dim > a->head_dim || a->rotary.dim % 2 != 0 ||
          !std::isfinite(a->rotary.theta) || a->rotary.theta <= 0 ||
          !std::isfinite(a->rotary.factor) || a->rotary.factor <= 0 ||
          a->sliding_window < 0) {
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
    if (a == nullptr) {
      return UnimplementedError("recurrent mixer execution is not implemented");
    }
    if (a->output_gate != components::OutputGate::kNone) {
      return UnimplementedError("gated attention output is not implemented");
    }
    if (a->projection_bias) {
      return UnimplementedError("biased projections are not implemented");
    }
    ops::AttentionParams params;
    params.query_heads = a->query_heads;
    params.kv_heads = a->kv_heads;
    params.head_dim = a->head_dim;
    params.scale = 1.0f;
    params.sliding_window = a->sliding_window;
    INFERX_RETURN_IF_ERROR(ops::ValidateAttentionGeometry(params));
    if (!std::holds_alternative<components::SwiGluConfig>(block.feed_forward)) {
      return UnimplementedError("expert feed-forward execution is not implemented");
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
    } else {
      const auto& g = std::get<components::GatedDeltaNetConfig>(block.mixer);
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
  attention_.emplace();
  mlp_.emplace();
  int64_t query_dim = 0;
  int64_t kv_dim = 0;
  int64_t query_heads = 0;
  for (const auto& block : config_.blocks) {
    if (const auto* a = std::get_if<components::AttentionConfig>(&block.mixer)) {
      query_heads = std::max(query_heads, a->query_heads);
      query_dim = std::max<int64_t>(query_dim, a->query_heads * a->head_dim);
      kv_dim = std::max<int64_t>(kv_dim, a->kv_heads * a->head_dim);
    }
    if (const auto* dense = std::get_if<components::SwiGluConfig>(&block.feed_forward)) {
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
  INFERX_ASSIGN_OR_RETURN(mixed_, alloc2(dims.hidden_size));
  INFERX_ASSIGN_OR_RETURN(attention_->query, alloc_flat(query_dim));
  INFERX_ASSIGN_OR_RETURN(attention_->key, alloc_flat(kv_dim));
  INFERX_ASSIGN_OR_RETURN(attention_->value, alloc_flat(kv_dim));
  INFERX_ASSIGN_OR_RETURN(attention_->attn_out, alloc_flat(query_dim));
  INFERX_ASSIGN_OR_RETURN(mlp_->gate, alloc_flat(max_intermediate_));
  INFERX_ASSIGN_OR_RETURN(mlp_->up, alloc_flat(max_intermediate_));
  // Attention always runs the fused QKV projection, and packed gate/up rows
  // can be wider; one buffer serves both since they never overlap in time.
  INFERX_ASSIGN_OR_RETURN(packed_projection_,
      alloc_flat(std::max(query_dim + 2 * kv_dim, 2 * max_intermediate_)));
  INFERX_ASSIGN_OR_RETURN(attention_->plan.plan,
      Tensor::Empty(DataType::kInt32, Shape({3 * rows + 1}), device));
  attention_->plan.prefill_tile_rows = prefill_tile_rows_;
  if (enable_split_decode_) {
    constexpr int batch = ops::FlashDecodeWorkspace::kMaxBatch;
    constexpr int tiles = batch * ops::FlashDecodeWorkspace::kPartitions;
    INFERX_ASSIGN_OR_RETURN(auto plan,
        Tensor::Empty(DataType::kInt32, Shape({3 * tiles + batch + 2}), device));
    INFERX_ASSIGN_OR_RETURN(auto values,
        Tensor::Empty(DataType::kBFloat16, Shape({tiles * query_dim}), device));
    INFERX_ASSIGN_OR_RETURN(auto scores,
        Tensor::Empty(DataType::kFloat32, Shape({tiles * query_heads}), device));
    attention_->plan.decode = ops::FlashDecodeWorkspace{std::move(plan), std::move(values),
                                                          std::move(scores)};
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
  if (input.embeddings.has_value()) {
    if (input.embeddings->Rank() != 2 || input.embeddings->Dim(0) != rows ||
        input.embeddings->Dim(1) != config_.model.hidden_size ||
        input.embeddings->Device() != ctx.device()) {
      return InvalidArgumentError("invalid prepared decoder embeddings");
    }
  } else if (input.token_ids.Rank() != 1 ||
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
  INFERX_ASSIGN_OR_RETURN(Tensor hidden, hidden_->Slice(0, rows));
  if (input.embeddings.has_value()) {
    INFERX_RETURN_IF_ERROR(ctx.runtime().CopyAsync(
        hidden.Data(), input.embeddings->Data(), hidden.NBytes(), CopyKind::kDeviceToDevice,
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
  INFERX_RETURN_IF_ERROR(ops::BeginAttentionStep(ctx, attention_batch.kv_indptr,
      attention_batch.last_page_len, state.paged_kv->block_size(), rows,
      attention_batch.num_seqs, attention_->plan));
  for (size_t i = 0; i < config_.blocks.size(); ++i) {
    const std::string prefix = trace.enabled() ? "layer_" + std::to_string(i) + "." : "";
    const auto& layer = config_.blocks[i];
    const auto& weights = weights_.blocks[i];
    INFERX_ASSIGN_OR_RETURN(Tensor normed, normed_->Slice(0, rows));
    ops::RMSNormConfig norm{layer.norm.eps, layer.norm.plus_one, !layer.norm.plus_one};
    if (i == 0) {
      INFERX_RETURN_IF_ERROR(ops::RmsNorm(ctx, hidden, weights.input_norm, normed, norm));
    } else {
      INFERX_ASSIGN_OR_RETURN(Tensor previous_mixed, mixed_->Slice(0, rows));
      INFERX_RETURN_IF_ERROR(ops::AddRmsNorm(ctx, previous_mixed, hidden,
                                            weights.input_norm, normed, norm));
    }
    if (trace.enabled()) trace.Write(prefix + "input_norm", normed);

    const auto& a = std::get<components::AttentionConfig>(layer.mixer);
    INFERX_ASSIGN_OR_RETURN(Tensor mixed, mixed_->Slice(0, rows));
    INFERX_RETURN_IF_ERROR(components::RunAttention(
        a, weights.mixer, normed, layer.norm.eps, attention_batch,
        std::get<PagedKvState>(state.layers[i]), *state.paged_kv, *attention_,
        packed_projection_.has_value() ? &*packed_projection_ : nullptr, ctx, &trace, prefix,
        mixed));

    // Feed-forward: norm, SwiGLU, project back; residual is fused into the
    // next layer's normalization.
    INFERX_RETURN_IF_ERROR(ops::AddRmsNorm(ctx, mixed, hidden, weights.post_mixer_norm, normed, norm));
    if (trace.enabled()) trace.Write(prefix + "post_norm", normed);
    if (trace.enabled()) trace.Write(prefix + "residual", hidden);
    INFERX_RETURN_IF_ERROR(components::RunFeedForward(
        layer.feed_forward, weights.feed_forward, normed, *mlp_,
        packed_projection_.has_value() ? &*packed_projection_ : nullptr, ctx, &trace, prefix,
        mixed));
  }

  INFERX_ASSIGN_OR_RETURN(Tensor final_rows, normed_->Slice(0, rows));
  INFERX_ASSIGN_OR_RETURN(Tensor last_mixed, mixed_->Slice(0, rows));
  INFERX_RETURN_IF_ERROR(ops::AddRmsNorm(
      ctx, last_mixed, hidden, weights_.final_norm, final_rows,
      ops::RMSNormConfig{config_.final_norm.eps, config_.final_norm.plus_one,
                        !config_.final_norm.plus_one}));
  if (trace.enabled()) trace.Write("final_norm", final_rows);
  return final_rows;
}

}  // namespace inferx::causal
