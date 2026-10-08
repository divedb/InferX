#include "inferx/models/layers/decoder.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>

#include "inferx/core/device_runtime.h"
#include "inferx/core/shape.h"
#include "inferx/ops/attention.h"
#include "inferx/ops/elementwise.h"
#include "inferx/ops/flash_attention.h"
#include "inferx/ops/gather.h"

namespace inferx::models::layers {

Status RunFeedForward(const std::variant<GatedMlpConfig, MoeConfig>& config,
                      const std::variant<GatedMlpWeights, MoeWeights>& weights,
                      const Tensor& normed, GatedMlpWorkspace& mlp_ws, MoeWorkspace& moe_ws,
                      Tensor* packed_buffer, ops::OpContext& ctx, Tensor& mixed_out) {
  if (const auto* dense = std::get_if<GatedMlpConfig>(&config)) {
    return RunGatedMlp(*dense, std::get<GatedMlpWeights>(weights), normed, mlp_ws, packed_buffer,
                       ctx, mixed_out);
  }
  return RunMoe(std::get<MoeConfig>(config), std::get<MoeWeights>(weights), normed, moe_ws,
                mlp_ws, packed_buffer, ctx, mixed_out);
}

Status StackConfig::Validate() const {
  if (hidden_size <= 0 || vocab_size <= 0 || blocks.empty()) {
    return InvalidArgumentError("decoder dimensions and layer count disagree");
  }
  if (!std::isfinite(final_norm.eps) || final_norm.eps < 0) {
    return InvalidArgumentError("invalid final normalization epsilon");
  }
  for (const auto& block : blocks) {
    if (!std::isfinite(block.norm.eps) || block.norm.eps < 0) {
      return InvalidArgumentError("invalid block normalization epsilon");
    }
    if (const auto* m = std::get_if<MlaConfig>(&block.mixer)) {
      if (m->query_heads <= 0 || m->q_lora_rank <= 0 || m->kv_lora_rank <= 0 ||
          m->qk_nope_head_dim <= 0 || m->qk_rope_head_dim <= 0 || m->v_head_dim <= 0 ||
          m->v_head_dim > m->head_dim() || m->rotary.rotary_dim != m->qk_rope_head_dim) {
        return InvalidArgumentError("invalid multi-latent attention geometry");
      }
    } else if (const auto* a = std::get_if<AttentionConfig>(&block.mixer)) {
      if (a->query_heads <= 0 || a->kv_heads <= 0 || a->head_dim <= 0 ||
          a->query_heads % a->kv_heads != 0 || a->rotary.rotary_dim <= 0 ||
          a->rotary.rotary_dim > a->head_dim || a->rotary.rotary_dim % 2 != 0 ||
          !std::isfinite(a->rotary.theta) || a->rotary.theta <= 0 ||
          !std::isfinite(a->rotary.scaling.factor) || a->rotary.scaling.factor <= 0 ||
          a->sliding_window < 0) {
        return InvalidArgumentError("invalid attention geometry or rotary configuration");
      }
    } else {
      const auto& g = std::get<GatedDeltaNetConfig>(block.mixer);
      if (g.key_heads <= 0 || g.value_heads <= 0 || g.key_dim <= 0 || g.value_dim <= 0 ||
          g.value_heads % g.key_heads != 0 || g.conv_kernel_size <= 0) {
        return InvalidArgumentError("invalid Gated DeltaNet geometry");
      }
    }
    if (const auto* dense = std::get_if<GatedMlpConfig>(&block.feed_forward)) {
      if (dense->intermediate_size <= 0) return InvalidArgumentError("invalid gated MLP width");
    } else {
      const auto& moe = std::get<MoeConfig>(block.feed_forward);
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

Status StackConfig::ValidateExecutable() const {
  for (const auto& block : blocks) {
    const auto* a = std::get_if<AttentionConfig>(&block.mixer);
    if (a == nullptr && !std::holds_alternative<MlaConfig>(block.mixer) &&
        !std::holds_alternative<GatedDeltaNetConfig>(block.mixer)) {
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
    if (const auto* moe = std::get_if<MoeConfig>(&block.feed_forward)) {
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

std::vector<LayerStateSpec> StackConfig::StateRequirements() const {
  std::vector<LayerStateSpec> specs;
  specs.reserve(blocks.size());
  for (const auto& block : blocks) {
    if (const auto* a = std::get_if<AttentionConfig>(&block.mixer)) {
      KvLayout layout;
      layout.kv_heads = a->kv_heads;
      layout.head_dim = a->head_dim;
      layout.dtype = DataType::kBFloat16;
      specs.push_back(PagedKvStateSpec{layout});
    } else if (const auto* m = std::get_if<MlaConfig>(&block.mixer)) {
      // Decompressed MLA caches per-head K and zero-padded V at head_dim.
      specs.push_back(PagedKvStateSpec{KvLayout{2, m->query_heads, m->head_dim(),
                                                DataType::kBFloat16}});
    } else {
      const auto& g = std::get<GatedDeltaNetConfig>(block.mixer);
      specs.push_back(RecurrentStateSpec{g.key_heads, g.value_heads, g.key_dim, g.value_dim,
                                         g.conv_kernel_size});
    }
  }
  return specs;
}

Status ShardStackForRank(StackConfig& config, const ParallelConfig& parallel) {
  INFERX_RETURN_IF_ERROR(parallel.Validate());
  const bool sharded = parallel.tensor_parallel_size > 1;
  for (auto& block : config.blocks) {
    auto* a = std::get_if<AttentionConfig>(&block.mixer);
    if (a != nullptr) {
      INFERX_ASSIGN_OR_RETURN(auto geometry, ShardQkv(*a, parallel));
      a->query_heads = geometry.query_heads;
      a->kv_heads = geometry.kv_heads;
      continue;
    }
    if (sharded) {
      return UnimplementedError(
          "MLA and recurrent mixers do not shard across tensor-parallel ranks yet");
    }
  }
  if (!sharded) return OkStatus();
  for (auto& block : config.blocks) {
    if (auto* dense = std::get_if<GatedMlpConfig>(&block.feed_forward)) {
      INFERX_ASSIGN_OR_RETURN(auto shard, ShardDim(dense->intermediate_size, parallel));
      dense->intermediate_size = shard.size;
    } else if (auto* moe = std::get_if<MoeConfig>(&block.feed_forward)) {
      INFERX_ASSIGN_OR_RETURN(auto shard, ShardDim(moe->intermediate_size, parallel));
      moe->intermediate_size = shard.size;
      if (moe->shared_intermediate_size > 0) {
        INFERX_ASSIGN_OR_RETURN(auto shared, ShardDim(moe->shared_intermediate_size, parallel));
        moe->shared_intermediate_size = shared.size;
      }
    }
  }
  return OkStatus();
}

DecoderLayer::DecoderLayer(const DecoderLayerConfig& config, DecoderLayerWeights weights)
    : input_norm_(config.norm, std::move(weights.input_norm)),
      ffn_norm_(config.norm, std::move(weights.post_mixer_norm)),
      mixer_out_norm_(MakeOutNorm(config.residual, config.mixer_out_norm,
                                  std::move(weights.mixer_out_norm))),
      feed_forward_out_norm_(MakeOutNorm(config.residual, config.feed_forward_out_norm,
                                         std::move(weights.feed_forward_out_norm))),
      mixer_(config.mixer),
      feed_forward_(config.feed_forward),
      mixer_weights_(std::move(weights.mixer)),
      feed_forward_weights_(std::move(weights.feed_forward)),
      norm_eps_(config.norm.eps),
      output_norm_residual_(config.residual == ResidualStyle::kOutputNorm) {}

std::optional<RmsNorm> DecoderLayer::MakeOutNorm(ResidualStyle residual, const NormConfig& config,
                                                 std::optional<Tensor> weight) {
  if (residual != ResidualStyle::kOutputNorm || !weight.has_value()) {
    return std::nullopt;
  }
  return RmsNorm(config, std::move(*weight));
}

Status DecoderLayer::Forward(bool first, Tensor& hidden, Tensor& normed, Tensor& mixed,
                             const AttentionBatch& batch, const LayerState& state,
                             const KvBlockPool* pool, const RecurrentStatePool* recurrent,
                             AttentionWorkspace& attention_ws, MlaWorkspace& mla_ws,
                             GdnWorkspace& gdn_ws, GatedMlpWorkspace& mlp_ws,
                             MoeWorkspace& moe_ws, Tensor& packed, ops::OpContext& ctx,
                             dist::CommBackend& comm) const {
  if (first) {
    INFERX_RETURN_IF_ERROR(input_norm_.Forward(ctx, hidden, normed));
  } else {
    INFERX_RETURN_IF_ERROR(input_norm_.AddForward(ctx, mixed, hidden, normed));
  }
  if (auto* gdn = std::get_if<GdnWeights>(&mixer_weights_)) {
    if (recurrent == nullptr) {
      return InvalidArgumentError("recurrent layer ran without a bound state pool");
    }
    INFERX_RETURN_IF_ERROR(RunGatedDeltaNet(
        std::get<GatedDeltaNetConfig>(mixer_), *gdn, normed, batch,
        std::get<RecurrentState>(state), *recurrent, gdn_ws, ctx, mixed));
  } else if (auto* mla = std::get_if<MlaWeights>(&mixer_weights_)) {
    INFERX_RETURN_IF_ERROR(RunMlaAttention(
        std::get<MlaConfig>(mixer_), *mla, normed, norm_eps_, batch,
        std::get<PagedKvState>(state), *pool, mla_ws, ctx, mixed));
  } else if (const auto* a = std::get_if<AttentionConfig>(&mixer_)) {
    INFERX_RETURN_IF_ERROR(
      RunAttention(*a, std::get<AttentionWeights>(mixer_weights_),
                   normed, norm_eps_, batch, std::get<PagedKvState>(state), *pool,
                   attention_ws, &packed, ctx, mixed));
  } else {
    return InvalidArgumentError("layer mixer has no weights");
  }
  // The mixer's output projection is row-parallel: fold the ranks'
  // partial sums before the result joins the residual stream (and
  // before Gemma's output-side norm, which applies to the full sum).
  INFERX_RETURN_IF_ERROR(comm.AllReduceSum(ctx, mixed));
  if (output_norm_residual_) {
    // Gemma sandwich: the mixer output is normalized before the residual
    // add, which the feed-forward-side norm then fuses.
    INFERX_RETURN_IF_ERROR(mixer_out_norm_->Forward(ctx, mixed, mixed));
  }
  INFERX_RETURN_IF_ERROR(ffn_norm_.AddForward(ctx, mixed, hidden, normed));
  INFERX_RETURN_IF_ERROR(RunFeedForward(feed_forward_, feed_forward_weights_,
                                        normed, mlp_ws, moe_ws, &packed, ctx, mixed));
  // Same folding for the feed-forward: down projections (dense, expert,
  // and shared) sum rank-local partials over the sharded input.
  INFERX_RETURN_IF_ERROR(comm.AllReduceSum(ctx, mixed));
  if (output_norm_residual_) {
    INFERX_RETURN_IF_ERROR(feed_forward_out_norm_->Forward(ctx, mixed, mixed));
  }
  return OkStatus();
}

DecoderStack::DecoderStack(StackConfig config, StackWeights weights, int max_tokens)
    : config_(std::move(config)),
      embedding_(std::move(weights.token_embedding), config_.vocab_size,
                 config_.embedding_scale),
      max_tokens_(max_tokens),
      final_norm_(config_.final_norm, std::move(weights.final_norm)) {
  // Account the weights while they are still enumerable, before the layer
  // objects take ownership.
  weights_bytes_ = embedding_.weight().NBytes() + final_norm_.weight().NBytes();
  const auto add = [this](const Tensor& t) { weights_bytes_ += t.NBytes(); };
  for (auto& block : weights.blocks) {
    add(block.input_norm);
    add(block.post_mixer_norm);
    if (block.mixer_out_norm.has_value()) add(*block.mixer_out_norm);
    if (block.feed_forward_out_norm.has_value()) add(*block.feed_forward_out_norm);
    std::visit(
        [&](auto& mixer) {
          using M = std::decay_t<decltype(mixer)>;
          if constexpr (std::is_same_v<M, AttentionWeights>) {
            add(mixer.packed_qkv);
            add(mixer.output.weight);
            if (mixer.query_norm.has_value()) add(*mixer.query_norm);
            if (mixer.key_norm.has_value()) add(*mixer.key_norm);
            if (mixer.qkv_bias.has_value()) add(*mixer.qkv_bias);
            if (mixer.output_bias.has_value()) add(*mixer.output_bias);
            if (mixer.sinks.has_value()) add(*mixer.sinks);
          } else if constexpr (std::is_same_v<M, MlaWeights>) {
            add(mixer.q_a);
            add(mixer.q_a_norm);
            add(mixer.q_b);
            add(mixer.k_rope);
            add(mixer.kv_latent);
            add(mixer.kv_a_norm);
            add(mixer.kv_b);
            add(mixer.output);
          } else {
            add(mixer.in_proj_qkvz);
            add(mixer.in_proj_ba);
            add(mixer.conv1d);
            add(mixer.a_log);
            add(mixer.dt_bias);
            add(mixer.norm_weight);
            add(mixer.out_proj);
          }
        },
        block.mixer);
    std::visit(
        [&](auto& ffn) {
          using F = std::decay_t<decltype(ffn)>;
          if constexpr (std::is_same_v<F, GatedMlpWeights>) {
            add(ffn.packed_gate_up);
            add(ffn.down.weight);
            if (ffn.packed_bias.has_value()) add(*ffn.packed_bias);
            if (ffn.down_bias.has_value()) add(*ffn.down_bias);
          } else {
            add(ffn.router);
            if (ffn.router_bias.has_value()) add(*ffn.router_bias);
            if (ffn.correction_bias.has_value()) add(*ffn.correction_bias);
            for (auto& expert : ffn.experts) {
              add(expert.packed_gate_up);
              add(expert.down.weight);
            }
            if (ffn.shared_expert.has_value()) {
              add(ffn.shared_expert->packed_gate_up);
              add(ffn.shared_expert->down.weight);
            }
            if (ffn.shared_expert_gate.has_value()) add(*ffn.shared_expert_gate);
          }
        },
        block.feed_forward);
  }
  // Experimental until full-model numerical parity is established. Read once
  // per model so graph capture and replay always use the same implementation.
  const char* flag = std::getenv("INFERX_EXPERIMENTAL_SPLIT_DECODE");
  enable_split_decode_ = flag != nullptr && std::string_view(flag) == "1";
  flag = std::getenv("INFERX_EXPERIMENTAL_PREFILL_TILE");
  if (flag != nullptr && std::string_view(flag) == "128") prefill_tile_rows_ = 128;
  layers_.reserve(config_.blocks.size());
  for (size_t i = 0; i < config_.blocks.size(); ++i) {
    layers_.emplace_back(config_.blocks[i], std::move(weights.blocks[i]));
  }
}

Status DecoderStack::Prepare(int max_tokens, int max_sequences, DeviceId device) {
  (void)max_sequences;  // Workspace is sized by the token budget alone today.
  if (workspace_ready_) {
    if (max_tokens > max_tokens_) {
      return InvalidArgumentError("execution limits exceed the prepared workspace capacity");
    }
    return OkStatus();
  }
  INFERX_RETURN_IF_ERROR(InitWorkspace(device));
  limits_tokens_ = max_tokens;
  limits_sequences_ = max_sequences;
  return OkStatus();
}

int64_t DecoderStack::weights_bytes() const { return weights_bytes_; }

int64_t DecoderStack::workspace_bytes() const { return workspace_bytes_; }

Status DecoderStack::InitWorkspace(DeviceId device) {
  workspace_bytes_ = 0;
  const auto account = [this](const Tensor& t) { workspace_bytes_ += t.NBytes(); };
  int64_t query_dim = 0;
  int64_t kv_dim = 0;
  int64_t query_heads = 0;
  bool any_gated = false;
  int64_t packed_width = 0;
  for (const auto& block : config_.blocks) {
    if (const auto* r = std::get_if<GatedDeltaNetConfig>(&block.mixer)) {
      max_gdn_proj_ = std::max(max_gdn_proj_,
                               2 * r->key_heads * r->key_dim + 2 * r->value_heads * r->value_dim);
      max_gdn_conv_ = std::max(max_gdn_conv_,
                               2 * r->key_heads * r->key_dim + r->value_heads * r->value_dim);
      max_gdn_value_ = std::max(max_gdn_value_, r->value_heads * r->value_dim);
      max_gdn_heads_ = std::max(max_gdn_heads_, r->value_heads);
    }
    if (const auto* m = std::get_if<MlaConfig>(&block.mixer)) {
      max_mla_heads_ = std::max(max_mla_heads_, m->query_heads);
      max_mla_head_dim_ = std::max(max_mla_head_dim_, m->head_dim());
      max_mla_q_lora_ = std::max(max_mla_q_lora_, m->q_lora_rank);
      max_mla_rope_ = std::max(max_mla_rope_, m->qk_rope_head_dim);
      max_mla_kv_lora_ = std::max(max_mla_kv_lora_, m->kv_lora_rank);
      max_mla_up_width_ =
          std::max(max_mla_up_width_, m->query_heads * (m->qk_nope_head_dim + m->v_head_dim));
    }
    if (const auto* a = std::get_if<AttentionConfig>(&block.mixer)) {
      query_heads = std::max(query_heads, a->query_heads);
      query_dim = std::max<int64_t>(query_dim, a->query_heads * a->head_dim);
      kv_dim = std::max<int64_t>(kv_dim, a->kv_heads * a->head_dim);
      const bool gated = a->output_gate == OutputGate::kSigmoid;
      any_gated = any_gated || gated;
      packed_width = std::max<int64_t>(
          packed_width, (gated ? 2 : 1) * a->query_heads * a->head_dim + 2 * a->kv_heads * a->head_dim);
    }
    if (const auto* dense = std::get_if<GatedMlpConfig>(&block.feed_forward)) {
      max_intermediate_ = std::max(max_intermediate_, dense->intermediate_size);
    }
    if (const auto* moe = std::get_if<MoeConfig>(&block.feed_forward)) {
      max_moe_experts_ = std::max(max_moe_experts_, moe->num_experts);
      max_experts_per_token_ = std::max(max_experts_per_token_, moe->experts_per_token);
      max_moe_intermediate_ = std::max(max_moe_intermediate_, moe->intermediate_size);
      max_shared_intermediate_ =
          std::max(max_shared_intermediate_, moe->shared_intermediate_size);
      // The shared expert runs through the dense gated-MLP path and its
      // activation buffer is the dense workspace's.
      max_intermediate_ = std::max(max_intermediate_, moe->shared_intermediate_size);
    }
  }
  // Variant workspaces are sized by the scan above; their buffers are
  // created only when a layer of that kind exists.
  has_moe_ = max_moe_experts_ > 0;
  has_mla_ = max_mla_heads_ > 0;
  has_gdn_ = max_gdn_proj_ > 0;

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
  INFERX_ASSIGN_OR_RETURN(hidden_, alloc2(config_.hidden_size));
  INFERX_ASSIGN_OR_RETURN(normed_, alloc2(config_.hidden_size));
  INFERX_ASSIGN_OR_RETURN(mixed_, alloc2(config_.hidden_size));
  INFERX_ASSIGN_OR_RETURN(attention_.query, alloc_flat(query_dim));
  INFERX_ASSIGN_OR_RETURN(attention_.key, alloc_flat(kv_dim));
  INFERX_ASSIGN_OR_RETURN(attention_.value, alloc_flat(kv_dim));
  INFERX_ASSIGN_OR_RETURN(attention_.attn_out, alloc_flat(query_dim));
  if (any_gated) {
    INFERX_ASSIGN_OR_RETURN(attention_.gate, alloc_flat(query_dim));
  }
  INFERX_ASSIGN_OR_RETURN(mlp_.gate, alloc_flat(max_intermediate_));
  if (has_mla_) {
    const auto alloc_mla_flat = [&](int64_t width) {
      return Tensor::Empty(DataType::kBFloat16, Shape({rows * width}), device);
    };
    INFERX_ASSIGN_OR_RETURN(mla_.q_lora, alloc_mla_flat(max_mla_q_lora_));
    INFERX_ASSIGN_OR_RETURN(mla_.kv_lora, alloc_mla_flat(max_mla_kv_lora_));
    INFERX_ASSIGN_OR_RETURN(mla_.k_rope, alloc_mla_flat(max_mla_rope_));
    INFERX_ASSIGN_OR_RETURN(mla_.kv_b, alloc_mla_flat(max_mla_up_width_));
    INFERX_ASSIGN_OR_RETURN(mla_.query, alloc_mla_flat(max_mla_heads_ * max_mla_head_dim_));
    INFERX_ASSIGN_OR_RETURN(mla_.key, alloc_mla_flat(max_mla_heads_ * max_mla_head_dim_));
    INFERX_ASSIGN_OR_RETURN(mla_.value, alloc_mla_flat(max_mla_heads_ * max_mla_head_dim_));
    INFERX_ASSIGN_OR_RETURN(mla_.attn_out, alloc_mla_flat(max_mla_heads_ * max_mla_head_dim_));
    INFERX_ASSIGN_OR_RETURN(mla_.plan.plan,
                            Tensor::Empty(DataType::kInt32, Shape({3 * rows + 1}), device));
  }
  if (has_gdn_) {
    const auto alloc_gdn_flat = [&](int64_t width) {
      return Tensor::Empty(DataType::kBFloat16, Shape({rows * width}), device);
    };
    INFERX_ASSIGN_OR_RETURN(gdn_.packed, alloc_gdn_flat(max_gdn_proj_));
    INFERX_ASSIGN_OR_RETURN(gdn_.conv_in, alloc_gdn_flat(max_gdn_conv_));
    INFERX_ASSIGN_OR_RETURN(gdn_.z, alloc_gdn_flat(max_gdn_value_));
    INFERX_ASSIGN_OR_RETURN(gdn_.ba, alloc_gdn_flat(2 * max_gdn_heads_));
    INFERX_ASSIGN_OR_RETURN(
        gdn_.beta, Tensor::Empty(DataType::kFloat32, Shape({rows * max_gdn_heads_}), device));
    INFERX_ASSIGN_OR_RETURN(
        gdn_.g, Tensor::Empty(DataType::kFloat32, Shape({rows * max_gdn_heads_}), device));
    INFERX_ASSIGN_OR_RETURN(gdn_.y, alloc_gdn_flat(max_gdn_value_));
  }
  if (has_moe_) {
    // Slots are (token, selected-expert) pairs; every buffer is indexed by
    // the dispatch slot so gather, expert GEMMs, and scatter agree.
    const int64_t slots = max_tokens_ * max_experts_per_token_;
    INFERX_ASSIGN_OR_RETURN(moe_.topk_indices,
                            Tensor::Empty(DataType::kInt32, Shape({slots}), device));
    INFERX_ASSIGN_OR_RETURN(moe_.topk_weights,
                            Tensor::Empty(DataType::kFloat32, Shape({slots}), device));
    INFERX_ASSIGN_OR_RETURN(moe_.counts,
                            Tensor::Empty(DataType::kInt32, Shape({max_moe_experts_}), device));
    INFERX_ASSIGN_OR_RETURN(moe_.offsets,
                            Tensor::Empty(DataType::kInt32, Shape({max_moe_experts_ + 1}), device));
    INFERX_ASSIGN_OR_RETURN(moe_.cursor,
                            Tensor::Empty(DataType::kInt32, Shape({max_moe_experts_}), device));
    INFERX_ASSIGN_OR_RETURN(moe_.token_rows,
                            Tensor::Empty(DataType::kInt32, Shape({slots}), device));
    INFERX_ASSIGN_OR_RETURN(moe_.weights_by_slot,
                            Tensor::Empty(DataType::kFloat32, Shape({slots}), device));
    INFERX_ASSIGN_OR_RETURN(moe_.gathered, alloc2_at(slots, config_.hidden_size));
    INFERX_ASSIGN_OR_RETURN(moe_.activated,
                            Tensor::Empty(DataType::kBFloat16,
                                          Shape({slots * max_moe_intermediate_}), device));
    INFERX_ASSIGN_OR_RETURN(moe_.expert_rows, alloc2_at(slots, config_.hidden_size));
    INFERX_ASSIGN_OR_RETURN(moe_.packed_gate_up,
                            Tensor::Empty(DataType::kBFloat16,
                                          Shape({slots * 2 * max_moe_intermediate_}), device));
    INFERX_ASSIGN_OR_RETURN(moe_.shared_out, alloc2_at(max_tokens_, config_.hidden_size));
    INFERX_ASSIGN_OR_RETURN(moe_.shared_gate, alloc_flat(max_tokens_));
  }
  // Attention runs the fused QKV projection (with a doubled query section
  // when gated), and packed gate/up rows can be wider; one buffer serves
  // both since they never overlap in time.
  INFERX_ASSIGN_OR_RETURN(packed_projection_,
                          alloc_flat(std::max(packed_width, 2 * max_intermediate_)));
  INFERX_ASSIGN_OR_RETURN(attention_.plan.plan,
                          Tensor::Empty(DataType::kInt32, Shape({3 * rows + 1}), device));
  attention_.plan.prefill_tile_rows = prefill_tile_rows_;
  if (enable_split_decode_) {
    constexpr int batch = ops::FlashDecodeWorkspace::kMaxBatch;
    constexpr int tiles = batch * ops::FlashDecodeWorkspace::kPartitions;
    INFERX_ASSIGN_OR_RETURN(
        auto plan, Tensor::Empty(DataType::kInt32, Shape({3 * tiles + batch + 2}), device));
    INFERX_ASSIGN_OR_RETURN(
        auto values, Tensor::Empty(DataType::kBFloat16, Shape({tiles * query_dim}), device));
    INFERX_ASSIGN_OR_RETURN(
        auto scores, Tensor::Empty(DataType::kFloat32, Shape({tiles * query_heads}), device));
    attention_.plan.decode =
        ops::FlashDecodeWorkspace{std::move(plan), std::move(values), std::move(scores)};
  }
  for (const auto* t : {&hidden_, &normed_, &mixed_, &packed_projection_}) {
    if (t->has_value()) account(**t);
  }
  for (const auto* t : {&attention_.query, &attention_.gate, &attention_.key,
                        &attention_.value, &attention_.attn_out}) {
    if (t->has_value()) account(**t);
  }
  for (const auto* t :
       {&mla_.q_lora, &mla_.kv_lora, &mla_.k_rope, &mla_.kv_b, &mla_.query, &mla_.key,
        &mla_.value, &mla_.attn_out}) {
    if (t->has_value()) account(**t);
  }
  for (const auto* t : {&gdn_.packed, &gdn_.conv_in, &gdn_.z, &gdn_.ba, &gdn_.beta,
                        &gdn_.g, &gdn_.y}) {
    if (t->has_value()) account(**t);
  }
  if (mlp_.gate.has_value()) account(*mlp_.gate);
  for (const auto* t :
       {&moe_.topk_indices, &moe_.topk_weights, &moe_.counts, &moe_.offsets, &moe_.cursor,
        &moe_.token_rows, &moe_.weights_by_slot, &moe_.gathered, &moe_.activated,
        &moe_.expert_rows, &moe_.packed_gate_up, &moe_.shared_out, &moe_.shared_gate}) {
    if (t->has_value()) account(**t);
  }
  ++workspace_generation_;
  workspace_ready_ = true;
  return OkStatus();
}

StatusOr<Tensor> DecoderStack::BeginForward(const Tensor& token_ids,
                                            const AttentionBatch& batch, ModelState& state,
                                            ops::OpContext& ctx, dist::CommBackend& comm) {
  const int rows = batch.num_tokens;
  if (rows <= 0 || rows > max_tokens_) {
    return InvalidArgumentError("decoder token count exceeds workspace capacity");
  }
  if (token_ids.Rank() != 1 || token_ids.Numel() != rows ||
      token_ids.GetDataType() != DataType::kInt32 || token_ids.Device() != ctx.Device()) {
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
    // Lazy fallback for direct callers that skipped Prepare.
    INFERX_RETURN_IF_ERROR(InitWorkspace(ctx.Device()));
    limits_tokens_ = std::max(limits_tokens_, max_tokens_);
    limits_sequences_ = std::max(limits_sequences_, 1);
  }

  // Embed tokens into the hidden workspace.
  INFERX_ASSIGN_OR_RETURN(Tensor hidden, hidden_->Slice(0, rows));
  INFERX_RETURN_IF_ERROR(embedding_.Embed(token_ids, hidden, ctx, comm));

  const auto& attention_batch = batch;
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
      state.paged_kv->BlockSize(), rows, attention_batch.num_seqs, attention_.plan));
  return hidden;
}

StatusOr<Tensor> DecoderStack::Forward(const Tensor& token_ids, const AttentionBatch& batch,
                                       ModelState& state, ops::OpContext& ctx,
                                       dist::CommBackend& comm) {
  INFERX_ASSIGN_OR_RETURN(Tensor hidden, BeginForward(token_ids, batch, state, ctx, comm));
  INFERX_ASSIGN_OR_RETURN(Tensor normed, normed_->Slice(0, batch.num_tokens));
  INFERX_ASSIGN_OR_RETURN(Tensor mixed, mixed_->Slice(0, batch.num_tokens));
  for (size_t i = 0; i < layers_.size(); ++i) {
    INFERX_RETURN_IF_ERROR(layers_[i].Forward(
        i == 0, hidden, normed, mixed, batch, state.layers[i], state.paged_kv,
        state.recurrent, attention_, mla_, gdn_, mlp_, moe_, *packed_projection_, ctx,
        comm));
  }
  INFERX_RETURN_IF_ERROR(final_norm_.AddForward(ctx, mixed, hidden, normed));
  return normed;
}

}  // namespace inferx::models::layers
