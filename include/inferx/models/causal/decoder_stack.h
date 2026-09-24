/// \file
/// \brief The generic causal decoder stack: embedding -> pre-norm layers ->
/// final norm. Phase-agnostic; prefill and decode are expressed entirely in
/// DecoderInput's ragged batch geometry.

#ifndef INFERX_MODELS_CAUSAL_DECODER_STACK_H_
#define INFERX_MODELS_CAUSAL_DECODER_STACK_H_

#include <optional>
#include <vector>

#include "inferx/core/status.h"
#include "inferx/core/tensor.h"
#include "inferx/models/components/attention.h"
#include "inferx/models/components/decoder_layer.h"
#include "inferx/models/components/mlp.h"
#include "inferx/models/components/norm.h"
#include "inferx/models/model.h"
#include "inferx/models/state.h"
#include "inferx/ops/execution_context.h"

namespace inferx::causal {

/// \brief Everything needed to build a decoder stack.
struct DecoderConfig {
  CheckpointConfig model;                        ///< Family-agnostic dimensions.
  components::NormConfig final_norm;             ///< Closing norm after the last layer.
  std::vector<components::DecoderLayerConfig> blocks;  ///< One entry per layer.

  /// \brief Checks internal consistency of dimensions, geometry, and variants.
  Status Validate() const;

  /// \brief Checks that every configured variant has an executable
  ///        implementation -- run at build time so an unexecutable model is
  ///        rejected before any weight is loaded, never at first Forward.
  Status ValidateExecutable() const;

  /// \brief Persistent state each layer needs, one entry per layer.
  std::vector<LayerStateSpec> StateRequirements() const;
};

/// \brief All stack weights, on the model's device.
struct DecoderWeights {
  Tensor token_embedding;  ///< [vocab, hidden]
  Tensor final_norm;       ///< [hidden]
  std::vector<components::DecoderLayerWeights> blocks;
};

/// \brief One step's stack input; prepared embeddings may replace token ids.
struct DecoderInput {
  Tensor token_ids;  ///< [num_tokens] int32, unless embeddings is set.
  std::optional<Tensor> embeddings;  ///< [num_tokens, hidden] prepared embeddings.
  AttentionBatch attention;  ///< Ragged batch and KV geometry.
};

/// \brief Executes a DecoderConfig over DecoderWeights. One concrete class:
/// topology is data, computation lives in the components it calls.
class DecoderStack final {
 public:
  DecoderStack(DecoderConfig config, DecoderWeights weights, int max_tokens);

  const CheckpointConfig& config() const { return config_.model; }
  std::vector<LayerStateSpec> StateRequirements() const {
    return config_.StateRequirements();
  }
  StatusOr<Tensor> Forward(const DecoderInput& input, ModelState& state,
                           ops::ExecutionContext& ctx);

 private:
  /// \brief Allocates the reusable activation workspace on first use.
  Status InitWorkspace(DeviceId device);

  DecoderConfig config_;
  DecoderWeights weights_;
  int max_tokens_;

  /// \brief Persistent activation workspace, sized by max_tokens_ and the
  /// widest per-layer geometry, allocated on first use. Forward returns views
  /// into these buffers; they stay valid until the next Forward call.
  bool workspace_ready_ = false;
  int64_t max_intermediate_ = 0;
  bool enable_split_decode_ = false;
  int prefill_tile_rows_ = 64;
  std::optional<Tensor> packed_projection_;
  std::optional<Tensor> hidden_, normed_, mixed_;
  std::optional<components::AttentionWorkspace> attention_;
  std::optional<components::MlpWorkspace> mlp_;
};

}  // namespace inferx::causal

#endif  // INFERX_MODELS_CAUSAL_DECODER_STACK_H_
