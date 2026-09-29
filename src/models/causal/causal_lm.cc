#include "inferx/models/causal/causal_lm.h"

#include <utility>

#include "absl/strings/str_cat.h"
#include "inferx/core/logging.h"
#include "inferx/models/components/parallel_linear.h"
#include "inferx/models/components/qkv_linear.h"
#include "inferx/models/loading/weight_loader.h"
#include "inferx/ops/gather.h"
#include "inferx/ops/linear.h"

namespace inferx::causal {
namespace {

/// \brief Rewrites the config rank-local: head counts, feed-forward
///        widths, and the embedding shard offset.
///
/// Loading keeps the total config (this rank's shard rows derive from it);
/// the executed DecoderStack receives the rewritten copy so workspaces, KV
/// layouts, and kernels see exactly this rank's dimensions. Under KV
/// replication the rank-local query/kv ratio still divides evenly, so the
/// rank-local config passes the same geometric checks as the total one.
Status ApplyParallelSharding(DecoderConfig& config, const ParallelConfig& parallel) {
  const bool sharded = parallel.tensor_parallel_size > 1;
  for (auto& block : config.blocks) {
    auto* a = std::get_if<components::AttentionConfig>(&block.mixer);
    if (a != nullptr) {
      INFERX_ASSIGN_OR_RETURN(auto geometry, components::ShardQkv(*a, parallel));
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
    if (auto* dense = std::get_if<components::SwiGluConfig>(&block.feed_forward)) {
      INFERX_ASSIGN_OR_RETURN(auto shard,
                              components::ShardDim(dense->intermediate_size, parallel));
      dense->intermediate_size = shard.size;
    } else if (auto* moe = std::get_if<components::MoeConfig>(&block.feed_forward)) {
      INFERX_ASSIGN_OR_RETURN(auto shard, components::ShardDim(moe->intermediate_size, parallel));
      moe->intermediate_size = shard.size;
      if (moe->shared_intermediate_size > 0) {
        INFERX_ASSIGN_OR_RETURN(
            auto shared, components::ShardDim(moe->shared_intermediate_size, parallel));
        moe->shared_intermediate_size = shared.size;
      }
    }
  }
  INFERX_ASSIGN_OR_RETURN(auto vocab, components::ShardDim(config.model.vocab_size, parallel));
  config.embedding_row_offset = vocab.begin;
  return OkStatus();
}

}  // namespace

StatusOr<Tensor> LanguageModelHead::Forward(const Tensor& hidden, const Tensor& rows,
                                            ops::ExecutionContext& ctx, dist::CommBackend& comm) {
  if (hidden.Rank() != 2 || rows.Rank() != 1 || weight.Rank() != 2 ||
      hidden.Dim(1) != weight.Dim(1) || rows.GetDataType() != DataType::kInt32 ||
      hidden.Device() != ctx.device() || rows.Device() != ctx.device() ||
      weight.Device() != ctx.device()) {
    return InvalidArgumentError("invalid language-model head inputs");
  }
  const int64_t count = rows.Numel();
  if (count == 0) return InvalidArgumentError("language-model head needs at least one row");
  const int64_t vocab = vocab_total > 0 ? vocab_total : weight.Dim(0);
  if (vocab != comm.size() * weight.Dim(0)) {
    return InvalidArgumentError("language-model head and communicator topology disagree");
  }
  if (!workspace_ready_ || rows_->Dim(0) < count) {
    // Reserve all sequence slots before any capture: later batch growth must
    // not invalidate the row buffer referenced by an earlier decode graph.
    const int64_t reserve = std::max<int64_t>(count, capacity);
    INFERX_ASSIGN_OR_RETURN(
        rows_,
        Tensor::Empty(DataType::kBFloat16, Shape({reserve, weight.Dim(1)}), ctx.device()));
    INFERX_ASSIGN_OR_RETURN(
        logits_,
        Tensor::Empty(DataType::kBFloat16, Shape({reserve, weight.Dim(0)}), ctx.device()));
    if (vocab != weight.Dim(0)) {
      INFERX_ASSIGN_OR_RETURN(
          full_logits_,
          Tensor::Empty(DataType::kBFloat16, Shape({reserve, vocab}), ctx.device()));
    }
    workspace_ready_ = true;
  }
  INFERX_ASSIGN_OR_RETURN(Tensor row_batch, rows_->Slice(0, count));
  INFERX_RETURN_IF_ERROR(ops::GatherRows(ctx, hidden, rows, row_batch));
  INFERX_ASSIGN_OR_RETURN(Tensor logits, logits_->Slice(0, count));
  INFERX_RETURN_IF_ERROR(ops::Linear(ctx, row_batch, weight, logits));
  // Vocab-parallel head: this rank computed [rows, shard] logits over its
  // slice of the vocabulary; the gather lays all shards side by side.
  Tensor full = logits;
  if (full_logits_.has_value()) {
    INFERX_ASSIGN_OR_RETURN(full, full_logits_->Slice(0, count));
  }
  INFERX_RETURN_IF_ERROR(comm.AllGatherLastDim(ctx, logits, full));
  return full;
}

StatusOr<PreparedCausalLM> PrepareCausalLM(models::LoadedCheckpoint& checkpoint,
                                           DecoderConfig config,
                                           const models::WeightNames& names,
                                           const models::WeightLayout& layout, DeviceId device,
                                           int max_tokens, int max_seqs,
                                           const ParallelConfig& parallel) {
  INFERX_RETURN_IF_ERROR(parallel.Validate());
  INFERX_RETURN_IF_ERROR(config.Validate());
  DecoderConfig rank_local = config;  // Execution copy; loading uses totals.
  INFERX_RETURN_IF_ERROR(ApplyParallelSharding(rank_local, parallel));
  INFERX_RETURN_IF_ERROR(rank_local.ValidateExecutable());
  if (max_tokens <= 0 || max_seqs <= 0) {
    return InvalidArgumentError("model capacities must be positive");
  }
  INFERX_ASSIGN_OR_RETURN(auto weights, LoadDecoderWeights(checkpoint.weights, config, names,
                                                           layout, parallel, device));
  const auto& mc = config.model;
  std::optional<Tensor> head_weight;
  if (mc.tie_word_embeddings) {
    head_weight = weights.token_embedding;
  } else {
    INFERX_ASSIGN_OR_RETURN(
        head_weight, LoadVocabShard(checkpoint.weights, std::string(names.head) + ".weight",
                                    mc.vocab_size, mc.hidden_size, parallel, device));
  }
  std::string mixer_summary = "recurrent";
  if (const auto* first_attention =
          std::get_if<components::AttentionConfig>(&rank_local.blocks.front().mixer)) {
    mixer_summary = absl::StrCat("q_heads=", first_attention->query_heads,
                                 " kv_heads=", first_attention->kv_heads,
                                 " head_dim=", first_attention->head_dim);
  } else if (const auto* mla =
                 std::get_if<components::MlaConfig>(&rank_local.blocks.front().mixer)) {
    mixer_summary = absl::StrCat("mla heads=", mla->query_heads, " head_dim=", mla->head_dim());
  }
  INFERX_LOG(INFO) << "loaded decoder: layers=" << mc.num_hidden_layers
                   << " hidden=" << mc.hidden_size << " vocab=" << mc.vocab_size << " "
                   << mixer_summary << " tp=" << parallel.tensor_parallel_size;
  return PreparedCausalLM{std::move(rank_local), std::move(weights), std::move(*head_weight)};
}

}  // namespace inferx::causal
