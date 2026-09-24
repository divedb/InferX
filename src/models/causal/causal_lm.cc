#include "inferx/models/causal/causal_lm.h"

#include <utility>

#include "inferx/core/logging.h"
#include "inferx/models/causal/weight_mapping.h"
#include "inferx/models/components/qkv_linear.h"
#include "inferx/ops/gather.h"
#include "inferx/ops/linear.h"

namespace inferx::causal {
namespace {

/// \brief Rewrites each attention block's head counts rank-local.
///
/// Loading keeps the total config (this rank's shard rows derive from it);
/// the executed DecoderStack receives the rewritten copy so workspaces, KV
/// layouts, and kernels see exactly this rank's heads. Under KV replication
/// the rank-local query/kv ratio still divides evenly, so the rank-local
/// config passes the same geometric checks as the total one.
Status ApplyParallelSharding(DecoderConfig& config, const ParallelConfig& parallel) {
  for (auto& block : config.blocks) {
    auto* a = std::get_if<components::AttentionConfig>(&block.mixer);
    if (a == nullptr) continue;
    INFERX_ASSIGN_OR_RETURN(auto geometry, components::ShardQkv(*a, parallel));
    a->query_heads = geometry.query_heads;
    a->kv_heads = geometry.kv_heads;
  }
  return OkStatus();
}

}  // namespace

StatusOr<Tensor> LanguageModelHead::Forward(const Tensor& hidden, const Tensor& rows,
                                            ops::ExecutionContext& ctx) {
  if (hidden.Rank() != 2 || rows.Rank() != 1 ||
      weight.Rank() != 2 || hidden.Dim(1) != weight.Dim(1) ||
      rows.GetDataType() != DataType::kInt32 || hidden.Device() != ctx.device() ||
      rows.Device() != ctx.device() || weight.Device() != ctx.device()) {
    return InvalidArgumentError("invalid language-model head inputs");
  }
  const int64_t count = rows.Numel();
  if (count == 0) return InvalidArgumentError("language-model head needs at least one row");
  if (!workspace_ready_ || rows_->Dim(0) < count) {
    // Reserve all sequence slots before any capture: later batch growth must
    // not invalidate the row buffer referenced by an earlier decode graph.
    const int64_t reserve = std::max<int64_t>(count, capacity);
    INFERX_ASSIGN_OR_RETURN(
        rows_, Tensor::Empty(DataType::kBFloat16, Shape({reserve, weight.Dim(1)}), ctx.device()));
    INFERX_ASSIGN_OR_RETURN(
        logits_, Tensor::Empty(DataType::kBFloat16, Shape({reserve, weight.Dim(0)}), ctx.device()));
    workspace_ready_ = true;
  }
  INFERX_ASSIGN_OR_RETURN(Tensor row_batch, rows_->Slice(0, count));
  INFERX_RETURN_IF_ERROR(ops::GatherRows(ctx, hidden, rows, row_batch));
  INFERX_ASSIGN_OR_RETURN(Tensor logits, logits_->Slice(0, count));
  INFERX_RETURN_IF_ERROR(ops::Linear(ctx, row_batch, weight, logits));
  return logits;
}

CausalLM::CausalLM(DecoderStack decoder, LanguageModelHead head, int max_seqs)
    : decoder_(std::move(decoder)), head_(std::move(head)), max_seqs_(max_seqs) {
  head_.capacity = max_seqs;
}

StatusOr<Tensor> CausalLM::Forward(const ModelInput& input, ModelState& state,
                                   ops::ExecutionContext& ctx) {
  if (input.attention.num_seqs <= 0 || input.attention.num_seqs > max_seqs_ ||
      input.logit_rows.Rank() != 1 || input.logit_rows.Numel() != input.attention.num_seqs ||
      input.logit_rows.GetDataType() != DataType::kInt32 ||
      input.logit_rows.Device() != ctx.device()) {
    return InvalidArgumentError("invalid requested language-model output rows");
  }
  DecoderInput decoder_input{input.token_ids, {}, input.attention};
  INFERX_ASSIGN_OR_RETURN(Tensor hidden, decoder_.Forward(decoder_input, state, ctx));
  return head_.Forward(hidden, input.logit_rows, ctx);
}

StatusOr<std::unique_ptr<Model>> BuildCausalLM(models::LoadedCheckpoint& checkpoint,
                                               DecoderConfig config, const CheckpointLayout& layout,
                                               DeviceId device, int max_tokens, int max_seqs,
                                               const ParallelConfig& parallel) {
  INFERX_RETURN_IF_ERROR(parallel.Validate());
  if (parallel.tensor_parallel_size > 1) {
    // Geometry and loading are shard-ready (QKV, merged gate/up, row-parallel
    // o_proj/down, vocab shards); forward is not: RowParallel partial sums
    // and sharded-vocab logits need cross-rank collectives.
    return UnimplementedError(
        "tensor-parallel execution awaits collectives; geometry and loading "
        "are shard-ready, forward is not");
  }
  INFERX_RETURN_IF_ERROR(config.Validate());
  DecoderConfig rank_local = config;  // Execution copy; loading uses totals.
  INFERX_RETURN_IF_ERROR(ApplyParallelSharding(rank_local, parallel));
  INFERX_RETURN_IF_ERROR(rank_local.ValidateExecutable());
  if (max_tokens <= 0 || max_seqs <= 0) {
    return InvalidArgumentError("model capacities must be positive");
  }
  INFERX_ASSIGN_OR_RETURN(auto weights,
                          LoadDecoderWeights(checkpoint.weights, config, layout, parallel, device));
  const auto& mc = config.model;
  std::optional<Tensor> head_weight;
  if (mc.tie_word_embeddings) {
    head_weight = weights.token_embedding;
  } else {
    INFERX_ASSIGN_OR_RETURN(head_weight,
                            LoadVocabShard(checkpoint.weights, layout.head_name, mc.vocab_size,
                                           mc.hidden_size, parallel, device));
  }
  LanguageModelHead head{std::move(*head_weight)};
  const auto& first_attention =
      std::get<components::AttentionConfig>(rank_local.blocks.front().mixer);
  INFERX_LOG(INFO) << "loaded decoder: layers=" << mc.num_hidden_layers
                   << " hidden=" << mc.hidden_size << " vocab=" << mc.vocab_size
                   << " q_heads=" << first_attention.query_heads
                   << " kv_heads=" << first_attention.kv_heads
                   << " head_dim=" << first_attention.head_dim
                   << " tp=" << parallel.tensor_parallel_size;
  DecoderStack decoder(std::move(rank_local), std::move(weights), max_tokens);
  return std::unique_ptr<Model>(new CausalLM(std::move(decoder), std::move(head), max_seqs));
}

}  // namespace inferx::causal
