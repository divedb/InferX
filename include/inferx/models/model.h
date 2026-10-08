#ifndef INFERX_MODELS_MODEL_H_
#define INFERX_MODELS_MODEL_H_

#include <memory>
#include <optional>

#include "absl/types/span.h"
#include "inferx/core/status.h"
#include "inferx/core/tensor.h"
#include "inferx/models/forward_context.h"
#include "inferx/models/load_options.h"
#include "inferx/models/model_info.h"
#include "inferx/models/requirements.h"

namespace inferx::models {

/// \brief Logical batch metadata for one flat token batch.
///
/// Device arrays are int32; host indptr spans need only outlive the Forward
/// call that consumes them. The model runner prepares this struct from its
/// scheduler's plan; models read it as raw execution geometry.
struct AttentionBatch {
  Tensor positions;      ///< [num_tokens] position of each token.
  Tensor batch_indices;  ///< [num_tokens] sequence index of each token.
  Tensor qo_indptr;      ///< [num_seqs + 1] query-token offsets per sequence.
  Tensor kv_indptr;      ///< [num_seqs + 1] cumulative KV block counts.
  Tensor kv_indices;     ///< Flattened block tables of all sequences.
  Tensor last_page_len;  ///< [num_seqs] tokens in each sequence's last block.
  absl::Span<const int32_t>
      host_qo_indptr;  ///< Required mirror of qo_indptr for planning; must match device data.
  absl::Span<const int32_t> host_kv_indptr;  ///< Mirror of kv_indptr, host.
  int num_tokens = 0;                        ///< Total scheduled tokens.
  int num_seqs = 0;                          ///< Sequences in the batch.
  /// [num_seqs] recurrent-state slot of each sequence; absent when the model
  /// has no recurrent layers.
  std::optional<Tensor> recurrent_indices;
};

/// \brief One step's model execution inputs, prepared by the model runner.
struct ModelInputs {
  Tensor token_ids;          ///< [num_tokens] ids to compute, flat batch.
  AttentionBatch attention;  ///< Attention geometry and KV block tables.
};

/// \brief Backbone output of Forward: the hidden state of every token in the
///        batch, [num_tokens, hidden], on the model's device.
///
/// The tensor borrows model workspace: it stays valid until the next
/// Forward or ComputeLogits call on the same model.
struct HiddenStates {
  Tensor hidden;
};

/// \brief Which hidden-state rows need logits: [num_seqs] int32 row indices
///        (each sequence's last token). May be empty when no request needs
///        an output this step (for example a mid-prompt prefill chunk);
///        ComputeLogits is then skipped entirely.
struct LogitSelection {
  Tensor rows;
};

/// \brief Vocabulary projection of the selected rows: [rows, vocab_total]
///        logits, enqueued on the context's stream.
///
/// Like HiddenStates, the tensor borrows model workspace reused by the next
/// call. Under tensor parallelism the vocabulary is sharded per rank and the
/// returned logits are the all-gathered full width.
struct Logits {
  Tensor values;
};

/// \brief Execution capacities a model prepares its workspace for.
struct ExecutionLimits {
  int max_tokens = 0;      ///< One step's flat token budget.
  int max_sequences = 0;   ///< Concurrently executing sequences.
  int max_logit_rows = 0;  ///< Rows ComputeLogits must serve at once.
};

/// \brief Device memory a model holds, per rank, in bytes.
struct MemoryUsage {
  int64_t weights_bytes = 0;    ///< Loaded parameters on device.
  int64_t workspace_bytes = 0;  ///< Activation and scratch buffers.
};

/// \brief Stable identity of the launch structure a prepared step uses.
enum class ExecutionPathId : uint8_t {
  kEager,     ///< Enqueue the model callable directly.
  kCaptured,  ///< Replay a captured graph of the same callable.
};

/// \brief Whether the prepared step may run under graph capture. A custom
///        model with only a valid eager implementation reports no support.
struct GraphEligibility {
  bool eligible = false;
};

/// \brief What the runner needs to dispatch one prepared step. Model-owned
///        concrete plans stay private to the model implementation.
struct StepExecutionInfo {
  ExecutionPathId path = ExecutionPathId::kEager;
  GraphEligibility graph;
  /// Increments when workspace is reallocated; invalidates captured graphs.
  uint64_t workspace_generation = 0;
};

/// \brief Task-specific execution boundary for causal language models.
///
/// Implementations own their architecture configuration, layers, weights,
/// and activation workspace. The contract is serialized execution: the
/// runner guarantees no concurrent Forward, and keeps inputs alive until
/// the context's stream completes. Output-buffer lifetimes are documented
/// on HiddenStates and Logits; graph-captured forwards require stable
/// capacities and capture-compatible operations.
class CausalLanguageModel {
 public:
  virtual ~CausalLanguageModel() = default;

  /// \brief Engine-facing facts (vocabulary, context length, stop ids).
  virtual const ModelInfo& info() const = 0;

  /// \brief Persistent state the runtime must allocate and bind per layer.
  virtual const ModelRequirements& requirements() const = 0;

  /// \brief Allocates every model and layer workspace for the configured
  ///        limits, initializes backend handles, and finalizes stable
  ///        storage. State-neutral: never advances request state.
  ///
  /// After Prepare, prepared eager execution performs no device
  /// allocation. The default is a no-op for models that need no workspace.
  virtual Status Prepare(const ExecutionLimits& limits, ForwardContext& context) {
    (void)limits;
    (void)context;
    return OkStatus();
  }

  /// \brief Device memory this model holds (weights plus workspace).
  virtual MemoryUsage memory_usage() const { return {}; }

  /// \brief Validates model-specific geometry for one step and refreshes
  ///        plans, without advancing persistent sequence state. Host
  ///        planning and metadata uploads belong here, outside capture;
  ///        the next Forward consumes the most recent preparation.
  ///
  /// The default performs basic geometry validation and reports an eager,
  /// graph-ineligible path.
  virtual StatusOr<StepExecutionInfo> PrepareStep(const ModelInputs& inputs,
                                                  const LogitSelection& selection,
                                                  ForwardContext& context);

  /// \brief Runs the backbone over the prepared batch, updating bound state.
  virtual StatusOr<HiddenStates> Forward(const ModelInputs& inputs,
                                         ForwardContext& context) = 0;

  /// \brief Projects selected hidden-state rows to [rows, vocab] logits.
  ///        Skipped by the runner when the selection is empty.
  virtual StatusOr<Logits> ComputeLogits(const HiddenStates& states,
                                         const LogitSelection& selection,
                                         ForwardContext& context) = 0;
};

/// \brief Loads one checkpoint directory behind the CausalLanguageModel
///        boundary, selecting the architecture through the registry.
///
/// Identity is resolved from `config.json` alone, so unsupported
/// architectures are rejected before any weight shard is mapped.
StatusOr<std::unique_ptr<CausalLanguageModel>> LoadModel(const ModelLoadOptions& options);

}  // namespace inferx::models

#endif  // INFERX_MODELS_MODEL_H_
