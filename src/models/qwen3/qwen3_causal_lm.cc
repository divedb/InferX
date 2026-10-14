#include "inferx/models/qwen3/qwen3_causal_lm.h"

#include <utility>

#include "inferx/ops/gather.h"

namespace inferx {
namespace models {
namespace qwen3 {

StatusOr<std::unique_ptr<Qwen3ForCausalLM>> Qwen3ForCausalLM::Create(
    const Qwen3Config& config, const ParallelConfig& parallel, DataType dtype,
    Device device) {
  const int64_t tp = parallel.tensor_parallel_size;
  INFERX_ASSIGN_OR_RETURN(
      VocabParallelEmbedding embed_tokens,
      VocabParallelEmbedding::Create(config.vocab_size, config.hidden_size, tp,
                                            parallel.tensor_parallel_rank, dtype, device));
  INFERX_ASSIGN_OR_RETURN(Qwen3Model model,
                          Qwen3Model::Create(config, parallel, dtype, device));
  INFERX_ASSIGN_OR_RETURN(Linear lm_head,
                          Linear::Create(config.hidden_size, config.vocab_size / tp,
                                         dtype, device));
  if (config.tie_word_embeddings) {
    // The tied head is the embedding table: share its storage instead of
    // holding a second copy, and bind the (possibly absent) checkpoint
    // name onto the same bytes.
    *lm_head.MutableWeight() = embed_tokens.Weight();
  }

  ModelSpec spec;
  spec.architecture = "Qwen3ForCausalLM";
  spec.vocab_size = config.vocab_size;
  spec.hidden_size = config.hidden_size;
  spec.num_layers = config.num_hidden_layers;
  spec.max_position_embeddings = config.max_position_embeddings;
  spec.eos_token_ids = config.eos_token_ids;
  spec.tie_word_embeddings = config.tie_word_embeddings;
  spec.dtype = dtype;
  spec.device = device;
  if (config.num_key_value_heads % tp != 0) {
    return InvalidArgumentError("kv heads (", config.num_key_value_heads,
                                ") must divide tensor-parallel size ", tp);
  }
  inferx::KvLayout layout;
  layout.kv_heads = config.num_key_value_heads / tp;
  layout.head_dim = config.head_dim;
  spec.state.paged_kv.assign(config.num_hidden_layers, layout);

  return std::unique_ptr<Qwen3ForCausalLM>(
      new Qwen3ForCausalLM(std::move(embed_tokens), std::move(model), std::move(lm_head),
                           config.tie_word_embeddings, tp,
                           parallel.tensor_parallel_rank, std::move(spec)));
}

Status Qwen3ForCausalLM::BuildWeightMap(WeightMap& weights) {
  INFERX_RETURN_IF_ERROR(weights.Add(
      "model.embed_tokens.weight",
      WeightBinding::ShardRows(embed_tokens_.MutableWeight(),
                               embed_tokens_.Weight().Dim(0), tp_rank_)));
  INFERX_RETURN_IF_ERROR(model_.BindWeights("model", weights));
  auto head = WeightBinding::ShardRows(lm_head_.MutableWeight(),
                                       lm_head_.Weight().Dim(0), tp_rank_);
  head.required = !tied_;  // Tied checkpoints may omit lm_head.weight.
  return weights.Add("lm_head.weight", std::move(head));
}


Status Qwen3ForCausalLM::LoadWeights(CheckpointReader& checkpoint) {
  return LoadModelWeights(*this, checkpoint);
}

Status Qwen3ForCausalLM::Forward(ForwardContext& ctx, const StepInputs& inputs,
                      Tensor& hidden) const {
  const int64_t tokens = inputs.token_ids->Dim(0);
  const int64_t hidden_size = spec_.hidden_size;
  const DataType dtype = spec_.dtype;

  const Tensor* embedded = nullptr;
  std::optional<Tensor> owned;
  if (inputs.input_embeds.has_value()) {
    embedded = &*inputs.input_embeds;
    if (embedded->Rank() != 2 || embedded->Dim(0) != tokens ||
        embedded->Dim(1) != hidden_size || embedded->GetDataType() != dtype ||
        embedded->GetDevice() != ctx.op.GetDevice()) {
      return InvalidArgumentError("input_embeds must be [", tokens, ", ", hidden_size,
                                  "] ", DataTypeName(dtype), " on the context device");
    }
  } else {
    INFERX_ASSIGN_OR_RETURN(
        owned, Tensor::Empty(dtype, Shape({tokens, hidden_size}), &ctx.scratch));
    INFERX_RETURN_IF_ERROR(
        embed_tokens_.Forward(ctx.op, ctx.comm, *inputs.token_ids, *owned));
    embedded = &*owned;
  }

  // The residual stream starts as the (un-copied) embedding output; the
  // first decoder layer reads it plain-normed.
  Tensor residual = *embedded;
  return model_.Forward(ctx, *inputs.positions, *embedded, residual, hidden);
}

Status Qwen3ForCausalLM::ComputeLogits(ForwardContext& ctx, const Tensor& hidden, const Tensor& rows,
                            Tensor& logits) const {
  const int64_t row_count = rows.Dim(0);
  INFERX_ASSIGN_OR_RETURN(
      Tensor sampled_hidden,
      Tensor::Empty(spec_.dtype, Shape({row_count, spec_.hidden_size}), &ctx.scratch));
  INFERX_RETURN_IF_ERROR(ops::GatherRows(ctx.op, hidden, rows, sampled_hidden));

  if (tp_size_ == 1) {
    return lm_head_.Forward(ctx.op, sampled_hidden, logits);
  }
  const int64_t vocab_shard = lm_head_.Weight().Dim(0);
  INFERX_ASSIGN_OR_RETURN(
      Tensor shard_logits,
      Tensor::Empty(spec_.dtype, Shape({row_count, vocab_shard}), &ctx.scratch));
  INFERX_RETURN_IF_ERROR(lm_head_.Forward(ctx.op, sampled_hidden, shard_logits));
  return ctx.comm.AllGatherLastDim(ctx.op, shard_logits, logits);
}

}  // namespace qwen3
}  // namespace models
}  // namespace inferx
