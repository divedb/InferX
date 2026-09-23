#pragma once

#include <memory>
#include <string>

#include "absl/types/span.h"
#include "inferx/models/state.h"
#include "inferx/core/device.h"
#include "inferx/core/status.h"
#include "inferx/core/tensor.h"
#include "inferx/models/checkpoint_config.h"
#include "inferx/ops/execution_context.h"
#include "inferx/ops/attention.h"

namespace inferx {

/// \brief Logical batch metadata for one flat token batch.
///
/// Device arrays are int32; host indptr spans need only outlive the Forward
/// call that consumes them. No scheduler types cross this boundary: the model
/// runner prepares this struct from a SchedulerOutput, and models read it as
/// raw execution geometry.
struct AttentionBatch {
  Tensor positions;      ///< [num_tokens] position of each token.
  Tensor batch_indices;  ///< [num_tokens] sequence index of each token.
  Tensor qo_indptr;      ///< [num_seqs + 1] query-token offsets per sequence.
  Tensor kv_indptr;      ///< [num_seqs + 1] cumulative KV block counts.
  Tensor kv_indices;     ///< Flattened block tables of all sequences.
  Tensor last_page_len;  ///< [num_seqs] tokens in each sequence's last block.
  absl::Span<const int32_t> host_qo_indptr;  ///< Required mirror of qo_indptr for planning; must match device data.
  absl::Span<const int32_t> host_kv_indptr;  ///< Mirror of kv_indptr, host.
  int num_tokens = 0;                        ///< Total scheduled tokens.
  int num_seqs = 0;                          ///< Sequences in the batch.
};

/// \brief One step's model execution inputs, prepared by the ModelRunner.
struct ModelInput {
  Tensor token_ids;          ///< [num_tokens] ids to compute, flat batch.
  AttentionBatch attention;  ///< Attention geometry and KV block tables.
  Tensor logit_rows;         ///< [num_seqs] hidden-state row per sequence
                             ///< that needs logits (its last token).
};

/// \brief Serving interface for token-generation models.
///
/// A Model owns its weights and defines its own forward flow over ops. It
/// knows nothing about requests, scheduling, or vendor APIs; the runner
/// serializes execution and keeps inputs alive until the context's stream
/// completes. Returned tensors borrow reusable model workspace and stay valid
/// until the next Forward call.
class Model {
 public:
  virtual ~Model() = default;

  // Opt in only when Forward uses stable capacity buffers and device metadata
  // for changing decode positions/block tables. Custom models default to eager.
  virtual bool SupportsCudaGraphs() const { return false; }

  /// \brief Loads a checkpoint, selecting the implementation by architecture.
  ///
  /// \param directory Checkpoint directory with `config.json` and weights.
  /// \param device    Device to place weights on.
  /// \param max_tokens Workspace capacity for one step's token batch.
  /// \param max_seqs   Workspace capacity for concurrently sampled sequences.
  /// \return           The model, or an error status.
  static StatusOr<std::unique_ptr<Model>> Load(const std::string& directory, DeviceId device,
                                               int max_tokens, int max_seqs,
                                               ops::AttentionBackend backend = ops::AttentionBackend::kFlashInfer);

  /// \brief The parsed checkpoint configuration.
  virtual const CheckpointConfig& config() const = 0;

  /// \brief Per-layer persistent state required by the model. The runner must
  /// validate these requirements before selecting its allocation strategy.
  virtual std::vector<LayerStateSpec> StateRequirements() const = 0;

  /// \brief Runs one forward pass over the prepared batch.
  ///
  /// Reads tokens and attention metadata from `input`, updates the supplied model state, and returns [num_seqs, vocab] logits for
  /// input.logit_rows, enqueued on the context's stream.
  ///
  /// \param input  Execution inputs from the model runner.
  /// \param state  Per-layer execution state owned by the runner.
  /// \param ctx    Execution context ordering the work.
  /// \return       [num_seqs, vocab] logits, or an error status.
  virtual StatusOr<Tensor> Forward(const ModelInput& input, ModelState& state,
                                   ops::ExecutionContext& ctx) = 0;
};

}  // namespace inferx
