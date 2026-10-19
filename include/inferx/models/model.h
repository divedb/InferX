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
#include "inferx/dist/shard.h"
#include "inferx/models/model_spec.h"
#include "inferx/models/safetensors.h"
#include "inferx/models/weight_loader.h"
#include "inferx/ops/attention/op.h"
#include "inferx/ops/attention/runtime.h"
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
  ForwardContext(ops::OpContext& op, const attn::StepContext& step, CommBackend& comm,
                 Allocator& scratch, std::span<const attn::AttentionOp> attention)
      : op(op), step(step), comm(comm), scratch(scratch), attention(attention) {}

  /// Device, stream, and runtime the step executes on.
  ops::OpContext& op;
  /// The step's attention metadata, one GroupMeta per cache group.
  const attn::StepContext& step;
  /// The tensor-parallel group (identity at world size one).
  CommBackend& comm;
  /// Step-lifetime scratch; freed blocks are reused within the step.
  Allocator& scratch;

  /// The layer's attention executor, built by the runner's
  /// attn::AttentionRuntime at init (docs/attention.md §6.1).
  [[nodiscard]] const attn::AttentionOp& Attention(int64_t layer) const {
    return attention[static_cast<size_t>(layer)];
  }

  /// Index of the decoder layer currently executing; the tower sets it so
  /// attention modules address the right cache region.
  int64_t layer = 0;

 private:
  std::span<const attn::AttentionOp> attention;
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

  /// \brief The model's static description (device, vocab, eos, cache
  ///        requirements); assembled by the family config.
  virtual const ModelSpec& GetModelSpec() const = 0;

  /// \brief One attention spec per attention layer, in tower order; the
  ///        runner builds its attn::AttentionRuntime from these.
  virtual std::vector<attn::AttentionSpec> GetAttentionSpecs() const = 0;

  /// \brief Declarative parameter manifest for sharded loading
  ///        (docs/executor.md §7.1): global shapes plus ShardSpecs, one
  ///        entry per checkpoint tensor the model consumes. LocalShape
  ///        against a ParallelContext is the single source of truth for
  ///        local extents; `dst` is engaged wherever the parameter maps
  ///        1:1 to a layer tensor (fused-buffer components leave it empty
  ///        -- their placement stays model-side). Returned spans reference
  ///        storage owned by the model and live as long as it does.
  virtual std::vector<ParamDecl> DescribeParams() const = 0;

  virtual Status LoadWeights(CheckpointReader& checkpoint) = 0;

  virtual Status Forward(ForwardContext& ctx, const StepInputs& inputs,
                         Tensor& hidden) const = 0;

  virtual Status ComputeLogits(ForwardContext& ctx, const Tensor& hidden, const Tensor& rows,
                               Tensor& logits) const = 0;
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
