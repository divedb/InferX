#include "inferx/models/decoder_lm.h"

#include <utility>

#include "absl/strings/str_cat.h"
#include "inferx/core/logging.h"
#include "inferx/models/configs/dims.h"
#include "inferx/models/loading/weight_loader.h"

namespace inferx::models {

DecoderLm::DecoderLm(Prepared prepared, int max_tokens, int max_seqs)
    : info_(std::move(prepared.info)),
      stack_(std::move(prepared.config), std::move(prepared.weights), max_tokens),
      head_(std::move(prepared.head)),
      max_seqs_(max_seqs) {
  head_.capacity = max_seqs;
  head_.vocab_total = stack_.config().vocab_size;
  requirements_.layers = stack_.StateRequirements();
}

Status DecoderLm::Prepare(const ExecutionLimits& limits, ForwardContext& context) {
  if (limits.max_tokens <= 0 || limits.max_sequences <= 0 || limits.max_logit_rows <= 0) {
    return InvalidArgumentError("execution limits must be positive");
  }
  INFERX_RETURN_IF_ERROR(
      stack_.Prepare(limits.max_tokens, limits.max_sequences, context.ops.Device()));
  head_.capacity = std::max(head_.capacity, limits.max_logit_rows);
  INFERX_RETURN_IF_ERROR(head_.Prepare(limits.max_logit_rows, context.ops.Device()));
  return OkStatus();
}

MemoryUsage DecoderLm::memory_usage() const {
  MemoryUsage usage;
  usage.weights_bytes = stack_.weights_bytes() + head_.weight.NBytes();
  // A tied head aliases the token embedding; count the storage once.
  if (head_.weight.Bytes() == stack_.embedding_weight().Bytes()) {
    usage.weights_bytes -= head_.weight.NBytes();
  }
  usage.workspace_bytes = stack_.workspace_bytes() + head_.workspace_bytes();
  return usage;
}

StatusOr<StepExecutionInfo> DecoderLm::PrepareStep(const ModelInputs& inputs,
                                                   const LogitSelection& selection,
                                                   ForwardContext& context) {
  {
    auto validated = CausalLanguageModel::PrepareStep(inputs, selection, context);
    INFERX_RETURN_IF_ERROR(validated.status());
  }
  const auto& batch = inputs.attention;
  if (batch.num_tokens > stack_.limits_tokens() ||
      batch.num_seqs > stack_.limits_sequences() ||
      selection.rows.Numel() > stack_.limits_logit_rows()) {
    return InvalidArgumentError("step exceeds the prepared execution limits");
  }
  StepExecutionInfo info;
  info.workspace_generation = stack_.workspace_generation();
  // Graph eligibility: the shared stack is prepared and the device can
  // capture. The runner further restricts this to uniform decode steps.
  info.graph.eligible = context.ops.Device().IsCuda() && stack_.prepared();
  return info;
}

StatusOr<HiddenStates> DecoderLm::Forward(const ModelInputs& inputs, ForwardContext& context) {
  const auto& batch = inputs.attention;
  if (batch.num_seqs <= 0 || batch.num_seqs > max_seqs_) {
    return InvalidArgumentError("batch exceeds the model's sequence capacity");
  }
  INFERX_ASSIGN_OR_RETURN(Tensor hidden,
                          stack_.Forward(inputs.token_ids, batch, context.state, context.ops,
                                         context.comm));
  return HiddenStates{std::move(hidden)};
}

StatusOr<Logits> DecoderLm::ComputeLogits(const HiddenStates& states, const LogitSelection& selection,
                                          ForwardContext& context) {
  INFERX_ASSIGN_OR_RETURN(
      Tensor logits,
      head_.Forward(states.hidden, selection.rows, context.ops, context.comm));
  return Logits{std::move(logits)};
}

StatusOr<DecoderLm::Prepared> PrepareAttentionDecoder(layers::StackConfig stack,
                                                      const WeightNames& names,
                                                      const WeightLayout& layout,
                                                      std::string_view architecture,
                                                      const ModelDims& dims,
                                                      const ModelLoadOptions& options) {
  // Factory step 2: reject unsupported load and execution policy before
  // any weight is read.
  if (options.dtype != "auto" && options.dtype != "bfloat16") {
    return UnimplementedError("unsupported requested dtype: ", options.dtype,
                              " (supported: auto, bfloat16)");
  }
  if (options.max_tokens <= 0 || options.max_seqs <= 0) {
    return InvalidArgumentError("model capacities must be positive");
  }
  INFERX_RETURN_IF_ERROR(options.parallel.Validate());
  INFERX_RETURN_IF_ERROR(stack.Validate());

  // Rank-local execution copy; loading below uses the totals.
  layers::StackConfig rank_local = stack;
  INFERX_RETURN_IF_ERROR(layers::ShardStackForRank(rank_local, options.parallel));
  INFERX_RETURN_IF_ERROR(rank_local.ValidateExecutable());

  // Factory steps 3-4: structure is declared by `stack`; load the mapped
  // weights and the head, tied or separate.
  INFERX_ASSIGN_OR_RETURN(auto checkpoint, Checkpoint::Open(options.directory));
  INFERX_ASSIGN_OR_RETURN(auto weights,
                          loading::LoadStackWeights(checkpoint, stack, names, layout,
                                                    options.parallel, options.device));
  StatusOr<Tensor> head_weight = [&]() -> StatusOr<Tensor> {
    if (dims.tie_word_embeddings) return weights.token_embedding;
    return loading::LoadVocabShard(checkpoint, absl::StrCat(names.head, ".weight"),
                                   dims.vocab_size, dims.hidden_size, options.parallel,
                                   options.device);
  }();
  INFERX_RETURN_IF_ERROR(head_weight.status());
  const auto unconsumed = checkpoint.UnconsumedTensors();
  if (!unconsumed.empty()) {
    INFERX_LOG(WARNING) << "checkpoint contains " << unconsumed.size()
                        << " tensors the model did not load (first: " << unconsumed.front()
                        << "); check the architecture's weight mapping";
  }

  std::string mixer_summary = "recurrent";
  if (const auto* first_attention =
          std::get_if<layers::AttentionConfig>(&rank_local.blocks.front().mixer)) {
    mixer_summary = absl::StrCat("q_heads=", first_attention->query_heads,
                                 " kv_heads=", first_attention->kv_heads,
                                 " head_dim=", first_attention->head_dim);
  } else if (const auto* mla =
                 std::get_if<layers::MlaConfig>(&rank_local.blocks.front().mixer)) {
    mixer_summary = absl::StrCat("mla heads=", mla->query_heads, " head_dim=", mla->head_dim());
  }
  INFERX_LOG(INFO) << "loaded " << architecture << ": layers=" << stack.blocks.size()
                   << " hidden=" << stack.hidden_size << " vocab=" << stack.vocab_size << " "
                   << mixer_summary << " tp=" << options.parallel.tensor_parallel_size;

  ModelInfo info;
  info.architecture = std::string(architecture);
  info.vocab_size = stack.vocab_size;
  info.max_position_embeddings = dims.max_position_embeddings;
  info.bos_token_id = dims.bos_token_id;
  info.eos_token_id = dims.eos_token_id;
  return DecoderLm::Prepared{std::move(info), std::move(rank_local), std::move(weights),
                             std::move(*head_weight)};
}

}  // namespace inferx::models
