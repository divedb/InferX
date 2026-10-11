#ifndef INFERX_MODELS_MODEL_H_
#define INFERX_MODELS_MODEL_H_

#include <cstdint>
#include <memory>
#include <optional>
#include <string>

#include "inferx/cache/kv_block_pool.h"
#include "inferx/core/allocator.h"
#include "inferx/core/device_runtime.h"
#include "inferx/core/status.h"
#include "inferx/core/stream.h"
#include "inferx/core/tensor.h"
#include "inferx/dist/comm.h"
#include "inferx/models/layers/attention.h"
#include "inferx/models/model_spec.h"
#include "inferx/models/safetensors.h"
#include "inferx/models/weight_loader.h"
#include "inferx/ops/flash_attention.h"
#include "inferx/ops/op_context.h"

namespace inferx {
namespace models {

/// \brief Everything one forward step borrows from the runtime.
///
/// A context is assembled per step by the model runner and passed down the
/// model's own call tree; it owns nothing. Models read the batch plan,
/// carve scratch from the step allocator, select their layer's cache views
/// through `layer`, and issue collectives through `comm`. Static weights
/// live in the model; everything request-specific lives here.
class ForwardContext {
 public:
  ForwardContext(ops::OpContext& op, const AttentionMetadata& attn,
                 const KvBlockPool& kv, CommBackend& comm, Allocator& scratch,
                 ops::AttentionPlanWorkspace& plan)
      : op(op), attn(attn), kv(kv), comm(comm), scratch(scratch), plan(plan) {}

  /// Device, stream, and runtime the step executes on.
  ops::OpContext& op;
  /// The step's batch plan, shared by every layer.
  const AttentionMetadata& attn;
  /// The paged KV pool; layers take their cache views from it.
  const KvBlockPool& kv;
  /// The tensor-parallel group (identity at world size one).
  CommBackend& comm;
  /// Step-lifetime scratch; freed blocks are reused within the step.
  Allocator& scratch;
  /// Attention planning state, shared across layers and steps.
  ops::AttentionPlanWorkspace& plan;

  /// Index of the decoder layer currently executing; the tower sets it so
  /// attention modules address the right cache region.
  int64_t layer = 0;
};

/// \brief One step's token inputs. Positions ride with the tokens because
///        both index the step's flat token stream; multimodal callers may
///        substitute `input_embeds` for the embedding lookup.
struct StepInputs {
  /// Members are optional only so the struct can be default-constructed
  /// and filled in place; ids and positions are present before Forward.

  /// [num_tokens] int32 token ids, batch order.
  std::optional<Tensor> token_ids;

  /// [num_tokens] int32 absolute position of each token in its sequence.
  std::optional<Tensor> positions;

  /// [num_tokens, hidden] precomputed embeddings; when present the model
  /// skips its embedding table (multimodal adapters).
  std::optional<Tensor> input_embeds;
};

/// \brief One concrete model architecture, ready to load weights and run.
///
/// A model is static after `LoadWeights`: configuration, weight storage,
/// and the execution graph expressed by `Forward`. Request state, KV
/// allocation, and scheduling live in the runtime and reach the model only
/// through the `ForwardContext`. Implementations are final classes per
/// family; this interface exists because the registry and runner are
/// family-agnostic.
///
/// The forward pass is split the way the runtime consumes it: `Forward`
/// runs the whole stack and returns final-norm hidden states for every
/// token; `ComputeLogits` projects only the sampled rows onto the
/// vocabulary, so the LM-head GEMM costs [rows, vocab], not
/// [tokens, vocab].
class Model {
 public:
  virtual ~Model() = default;

  /// \brief Engine-facing facts (shapes, eos ids, state requirements).
  virtual const ModelSpec& GetModelSpec() const = 0;

  /// \brief Loads every weight this rank needs from the opened checkpoint.
  ///
  /// The default staging path runs on the model's own device; on return
  /// every bound tensor is filled and the model is immutable.
  virtual Status LoadWeights(CheckpointReader& checkpoint) = 0;

  /// \brief Runs the stack: embed (or take `inputs.input_embeds`), decode
  ///        layers, final norm.
  ///
  /// \param ctx     The step's execution context.
  /// \param inputs  Tokens and positions for this step.
  /// \param hidden  [num_tokens, hidden_size] final-norm activations.
  virtual Status Forward(ForwardContext& ctx, const StepInputs& inputs,
                         Tensor& hidden) const = 0;

  /// \brief Projects selected hidden rows onto the vocabulary.
  ///
  /// \param hidden  [num_tokens, hidden] from `Forward`.
  /// \param rows    [sample_rows] int32 token-stream rows whose logits are
  ///                needed (one per sequence: its last new token).
  /// \param logits  [sample_rows, vocab] output, allocated by the caller.
  virtual Status ComputeLogits(ForwardContext& ctx, const Tensor& hidden,
                               const Tensor& rows, Tensor& logits) const = 0;

};

/// \brief Stages one binding map into the model's tensors.
///
/// Opens a staging stream on the model's device, uploads every span, and
/// synchronizes. Shared by `LoadWeights` implementations and the
/// `LoadPolicy` paths, which adjust the map before handing it over.
Status LoadWeightMapInto(Model& model, WeightMap& weights, CheckpointReader& checkpoint);

/// \brief Loads a model's weights through its binding map.
///
/// Builds the map and stages it; weights are ready when this returns OK.
Status LoadModelWeights(Model& model, CheckpointReader& checkpoint);

/// \brief Zero-fills every tensor in a model's binding map.
///
/// The `LoadPolicy::dummy_weights` path: shapes and routing are real, the
/// bytes are not. The fill is stream-ordered on the model's device.
Status FillModelWeightsWithZeros(Model& model);

}  // namespace models
}  // namespace inferx

#endif  // INFERX_MODELS_MODEL_H_
