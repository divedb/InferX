#include "inferx/models/causal/decoder_stack.h"

#include <cmath>
#include <cstdlib>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "inferx/core/device_runtime.h"
#include "inferx/core/shape.h"
#include "inferx/models/diagnostic_trace.h"
#include "inferx/models/state.h"
#include "inferx/ops/attention.h"
#include "inferx/ops/flash_attention.h"
#include "inferx/ops/gather.h"
#include "inferx/ops/rms_norm.h"

namespace inferx::causal {

DecoderWorkspace::DecoderWorkspace(DecoderConfig config, Tensor embedding, int max_tokens)
    : config_(std::move(config)), embedding_(std::move(embedding)), max_tokens_(max_tokens) {
  // Experimental until full-model numerical parity is established. Read once
  // per model so graph capture and replay always use the same implementation.
  const char* flag = std::getenv("INFERX_EXPERIMENTAL_SPLIT_DECODE");
  enable_split_decode_ = flag != nullptr && std::string_view(flag) == "1";
  flag = std::getenv("INFERX_EXPERIMENTAL_PREFILL_TILE");
  if (flag != nullptr && std::string_view(flag) == "128") prefill_tile_rows_ = 128;
}

Status DecoderWorkspace::InitWorkspace(DeviceId device) {
  const auto& dims = config_.model;
  attention_.emplace();
  mlp_.emplace();
  int64_t query_dim = 0;
  int64_t kv_dim = 0;
  int64_t query_heads = 0;
  bool any_gated = false;
  int64_t packed_width = 0;
  for (const auto& block : config_.blocks) {
    if (const auto* r = std::get_if<components::GatedDeltaNetConfig>(&block.mixer)) {
      max_gdn_proj_ = std::max(max_gdn_proj_,
                               2 * r->key_heads * r->key_dim + 2 * r->value_heads * r->value_dim);
      max_gdn_conv_ = std::max(max_gdn_conv_,
                               2 * r->key_heads * r->key_dim + r->value_heads * r->value_dim);
      max_gdn_value_ = std::max(max_gdn_value_, r->value_heads * r->value_dim);
      max_gdn_heads_ = std::max(max_gdn_heads_, r->value_heads);
    }
    if (const auto* m = std::get_if<components::MlaConfig>(&block.mixer)) {
      max_mla_heads_ = std::max(max_mla_heads_, m->query_heads);
      max_mla_head_dim_ = std::max(max_mla_head_dim_, m->head_dim());
      max_mla_q_lora_ = std::max(max_mla_q_lora_, m->q_lora_rank);
      max_mla_rope_ = std::max(max_mla_rope_, m->qk_rope_head_dim);
      max_mla_kv_lora_ = std::max(max_mla_kv_lora_, m->kv_lora_rank);
      max_mla_up_width_ =
          std::max(max_mla_up_width_, m->query_heads * (m->qk_nope_head_dim + m->v_head_dim));
    }
    if (const auto* a = std::get_if<components::AttentionConfig>(&block.mixer)) {
      query_heads = std::max(query_heads, a->query_heads);
      query_dim = std::max<int64_t>(query_dim, a->query_heads * a->head_dim);
      kv_dim = std::max<int64_t>(kv_dim, a->kv_heads * a->head_dim);
      const bool gated = a->output_gate == components::OutputGate::kSigmoid;
      any_gated = any_gated || gated;
      packed_width = std::max<int64_t>(
          packed_width, (gated ? 2 : 1) * a->query_heads * a->head_dim + 2 * a->kv_heads * a->head_dim);
    }
    if (const auto* dense = std::get_if<components::SwiGluConfig>(&block.feed_forward)) {
      max_intermediate_ = std::max(max_intermediate_, dense->intermediate_size);
    }
    if (const auto* moe = std::get_if<components::MoeConfig>(&block.feed_forward)) {
      max_moe_experts_ = std::max(max_moe_experts_, moe->num_experts);
      max_experts_per_token_ = std::max(max_experts_per_token_, moe->experts_per_token);
      max_moe_intermediate_ = std::max(max_moe_intermediate_, moe->intermediate_size);
      max_shared_intermediate_ =
          std::max(max_shared_intermediate_, moe->shared_intermediate_size);
      // The shared expert runs through the dense SwiGLU path and its
      // activation buffer is the dense workspace's.
      max_intermediate_ = std::max(max_intermediate_, moe->shared_intermediate_size);
    }
  }
  // Variant workspaces are sized by the scan above, so they are created
  // only after it runs.
  if (max_moe_experts_ > 0) moe_.emplace();
  if (max_mla_heads_ > 0) mla_.emplace();
  if (max_gdn_proj_ > 0) gdn_.emplace();

  const int64_t rows = max_tokens_;
  const auto alloc2 = [&](int64_t cols) {
    return Tensor::Empty(DataType::kBFloat16, Shape({rows, cols}), device);
  };
  const auto alloc_flat = [&](int64_t cols) {
    return Tensor::Empty(DataType::kBFloat16, Shape({rows * cols}), device);
  };
  const auto alloc2_at = [&](int64_t r, int64_t c) {
    return Tensor::Empty(DataType::kBFloat16, Shape({r, c}), device);
  };
  INFERX_ASSIGN_OR_RETURN(hidden_, alloc2(dims.hidden_size));
  INFERX_ASSIGN_OR_RETURN(normed_, alloc2(dims.hidden_size));
  INFERX_ASSIGN_OR_RETURN(mixed_, alloc2(dims.hidden_size));
  INFERX_ASSIGN_OR_RETURN(attention_->query, alloc_flat(query_dim));
  INFERX_ASSIGN_OR_RETURN(attention_->key, alloc_flat(kv_dim));
  INFERX_ASSIGN_OR_RETURN(attention_->value, alloc_flat(kv_dim));
  INFERX_ASSIGN_OR_RETURN(attention_->attn_out, alloc_flat(query_dim));
  if (any_gated) {
    INFERX_ASSIGN_OR_RETURN(attention_->gate, alloc_flat(query_dim));
  }
  INFERX_ASSIGN_OR_RETURN(mlp_->gate, alloc_flat(max_intermediate_));
  if (mla_.has_value()) {
    const auto alloc_mla_flat = [&](int64_t width) {
      return Tensor::Empty(DataType::kBFloat16, Shape({rows * width}), device);
    };
    INFERX_ASSIGN_OR_RETURN(mla_->q_lora, alloc_mla_flat(max_mla_q_lora_));
    INFERX_ASSIGN_OR_RETURN(mla_->kv_lora, alloc_mla_flat(max_mla_kv_lora_));
    INFERX_ASSIGN_OR_RETURN(mla_->k_rope, alloc_mla_flat(max_mla_rope_));
    INFERX_ASSIGN_OR_RETURN(mla_->kv_b, alloc_mla_flat(max_mla_up_width_));
    INFERX_ASSIGN_OR_RETURN(mla_->query, alloc_mla_flat(max_mla_heads_ * max_mla_head_dim_));
    INFERX_ASSIGN_OR_RETURN(mla_->key, alloc_mla_flat(max_mla_heads_ * max_mla_head_dim_));
    INFERX_ASSIGN_OR_RETURN(mla_->value, alloc_mla_flat(max_mla_heads_ * max_mla_head_dim_));
    INFERX_ASSIGN_OR_RETURN(mla_->attn_out, alloc_mla_flat(max_mla_heads_ * max_mla_head_dim_));
    INFERX_ASSIGN_OR_RETURN(mla_->plan.plan,
                            Tensor::Empty(DataType::kInt32, Shape({3 * rows + 1}), device));
  }
  if (gdn_.has_value()) {
    const auto alloc_gdn_flat = [&](int64_t width) {
      return Tensor::Empty(DataType::kBFloat16, Shape({rows * width}), device);
    };
    INFERX_ASSIGN_OR_RETURN(gdn_->packed, alloc_gdn_flat(max_gdn_proj_));
    INFERX_ASSIGN_OR_RETURN(gdn_->conv_in, alloc_gdn_flat(max_gdn_conv_));
    INFERX_ASSIGN_OR_RETURN(gdn_->z, alloc_gdn_flat(max_gdn_value_));
    INFERX_ASSIGN_OR_RETURN(gdn_->ba, alloc_gdn_flat(2 * max_gdn_heads_));
    INFERX_ASSIGN_OR_RETURN(
        gdn_->beta, Tensor::Empty(DataType::kFloat32, Shape({rows * max_gdn_heads_}), device));
    INFERX_ASSIGN_OR_RETURN(
        gdn_->g, Tensor::Empty(DataType::kFloat32, Shape({rows * max_gdn_heads_}), device));
    INFERX_ASSIGN_OR_RETURN(gdn_->y, alloc_gdn_flat(max_gdn_value_));
  }
  if (moe_.has_value()) {
    // Slots are (token, selected-expert) pairs; every buffer is indexed by
    // the dispatch slot so gather, expert GEMMs, and scatter agree.
    const int64_t slots = max_tokens_ * max_experts_per_token_;
    INFERX_ASSIGN_OR_RETURN(moe_->topk_indices,
                            Tensor::Empty(DataType::kInt32, Shape({slots}), device));
    INFERX_ASSIGN_OR_RETURN(moe_->topk_weights,
                            Tensor::Empty(DataType::kFloat32, Shape({slots}), device));
    INFERX_ASSIGN_OR_RETURN(moe_->counts,
                            Tensor::Empty(DataType::kInt32, Shape({max_moe_experts_}), device));
    INFERX_ASSIGN_OR_RETURN(moe_->offsets,
                            Tensor::Empty(DataType::kInt32, Shape({max_moe_experts_ + 1}), device));
    INFERX_ASSIGN_OR_RETURN(moe_->cursor,
                            Tensor::Empty(DataType::kInt32, Shape({max_moe_experts_}), device));
    INFERX_ASSIGN_OR_RETURN(moe_->token_rows,
                            Tensor::Empty(DataType::kInt32, Shape({slots}), device));
    INFERX_ASSIGN_OR_RETURN(moe_->weights_by_slot,
                            Tensor::Empty(DataType::kFloat32, Shape({slots}), device));
    INFERX_ASSIGN_OR_RETURN(moe_->gathered, alloc2_at(slots, dims.hidden_size));
    INFERX_ASSIGN_OR_RETURN(moe_->activated,
                            Tensor::Empty(DataType::kBFloat16,
                                          Shape({slots * max_moe_intermediate_}), device));
    INFERX_ASSIGN_OR_RETURN(moe_->expert_rows, alloc2_at(slots, dims.hidden_size));
    INFERX_ASSIGN_OR_RETURN(moe_->packed_gate_up,
                            Tensor::Empty(DataType::kBFloat16,
                                          Shape({slots * 2 * max_moe_intermediate_}), device));
    INFERX_ASSIGN_OR_RETURN(moe_->shared_out, alloc2_at(max_tokens_, dims.hidden_size));
    INFERX_ASSIGN_OR_RETURN(moe_->shared_gate, alloc_flat(max_tokens_));
  }
  // Attention runs the fused QKV projection (with a doubled query section
  // when gated), and packed gate/up rows can be wider; one buffer serves
  // both since they never overlap in time.
  INFERX_ASSIGN_OR_RETURN(packed_projection_,
                          alloc_flat(std::max(packed_width, 2 * max_intermediate_)));
  INFERX_ASSIGN_OR_RETURN(attention_->plan.plan,
                          Tensor::Empty(DataType::kInt32, Shape({3 * rows + 1}), device));
  attention_->plan.prefill_tile_rows = prefill_tile_rows_;
  if (enable_split_decode_) {
    constexpr int batch = ops::FlashDecodeWorkspace::kMaxBatch;
    constexpr int tiles = batch * ops::FlashDecodeWorkspace::kPartitions;
    INFERX_ASSIGN_OR_RETURN(
        auto plan, Tensor::Empty(DataType::kInt32, Shape({3 * tiles + batch + 2}), device));
    INFERX_ASSIGN_OR_RETURN(
        auto values, Tensor::Empty(DataType::kBFloat16, Shape({tiles * query_dim}), device));
    INFERX_ASSIGN_OR_RETURN(
        auto scores, Tensor::Empty(DataType::kFloat32, Shape({tiles * query_heads}), device));
    attention_->plan.decode =
        ops::FlashDecodeWorkspace{std::move(plan), std::move(values), std::move(scores)};
  }
  workspace_ready_ = true;
  return OkStatus();
}

StatusOr<Tensor> DecoderWorkspace::BeginForward(const DecoderInput& input, ModelState& state,
                                                ops::ExecutionContext& ctx,
                                                dist::CommBackend& comm, DiagnosticTrace& trace) {
  if (comm.size() <= 0 || comm.rank() < 0 || comm.rank() >= comm.size() ||
      config_.model.vocab_size != comm.size() * embedding_.Dim(0) ||
      config_.embedding_row_offset != comm.rank() * embedding_.Dim(0)) {
    return InvalidArgumentError("decoder weights and communicator topology disagree");
  }
  const int rows = input.attention.num_tokens;
  if (rows <= 0 || rows > max_tokens_) {
    return InvalidArgumentError("decoder token count exceeds workspace capacity");
  }
  if (input.embeddings.has_value()) {
    if (input.embeddings->Rank() != 2 || input.embeddings->Dim(0) != rows ||
        input.embeddings->Dim(1) != config_.model.hidden_size ||
        input.embeddings->Device() != ctx.device()) {
      return InvalidArgumentError("invalid prepared decoder embeddings");
    }
  } else if (input.token_ids.Rank() != 1 || input.token_ids.Numel() != rows ||
             input.token_ids.GetDataType() != DataType::kInt32 ||
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
          entry->pool_layer >= state.paged_kv->NumLayers()) {
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
    INFERX_RETURN_IF_ERROR(ctx.runtime().CopyAsync(hidden.Data(), input.embeddings->Data(),
                                                   hidden.NBytes(), CopyKind::kDeviceToDevice,
                                                   ctx.stream()));
  } else {
    const bool sharded = comm.size() > 1;
    if (sharded) {
      // Vocab-parallel lookup: ids outside this rank's shard gather as
      // zeros; the all-reduce folds the ranks' masked partials.
      INFERX_RETURN_IF_ERROR(
          ops::GatherRowsRange(ctx, embedding_, input.token_ids, hidden,
                               config_.embedding_row_offset));
    } else {
      INFERX_RETURN_IF_ERROR(ops::GatherRows(ctx, embedding_, input.token_ids, hidden));
    }
    INFERX_RETURN_IF_ERROR(comm.AllReduceSum(ctx, hidden));
    if (config_.embedding_scale != 1.0f) {
      // Gemma scales token embeddings by sqrt(hidden) in activation dtype.
      INFERX_RETURN_IF_ERROR(ops::MulScalar(ctx, hidden, config_.embedding_scale));
    }
  }

  const auto& attention_batch = input.attention;
  if (trace.enabled()) trace.Write("embedding", hidden);
  if (attention_batch.num_seqs <= 0 ||
      attention_batch.host_qo_indptr.size() !=
          static_cast<size_t>(attention_batch.num_seqs + 1) ||
      attention_batch.host_qo_indptr.front() != 0 ||
      attention_batch.host_qo_indptr.back() != rows) {
    return InvalidArgumentError("attention requires host query offsets matching the batch");
  }
  for (int seq = 0; seq < attention_batch.num_seqs; ++seq) {
    if (attention_batch.host_qo_indptr[seq + 1] <= attention_batch.host_qo_indptr[seq])
      return InvalidArgumentError("attention sequences must have positive query lengths");
  }
  INFERX_RETURN_IF_ERROR(ops::BeginAttentionStep(
      ctx, attention_batch.kv_indptr, attention_batch.last_page_len,
      state.paged_kv->BlockSize(), rows, attention_batch.num_seqs, attention_->plan));
  return hidden;
}

}  // namespace inferx::causal
