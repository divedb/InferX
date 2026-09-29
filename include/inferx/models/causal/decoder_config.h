/// \file
/// \brief The generic causal decoder stack: embedding -> pre-norm layers ->
/// final norm. Phase-agnostic; prefill and decode are expressed entirely in
/// DecoderInput's ragged batch geometry.

#ifndef INFERX_MODELS_CAUSAL_DECODER_CONFIG_H_
#define INFERX_MODELS_CAUSAL_DECODER_CONFIG_H_

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
  CheckpointConfig model;                              ///< Family-agnostic dimensions.
  components::NormConfig final_norm;                   ///< Closing norm after the last layer.
  /// Embedding multiplier applied after the gather (Gemma: sqrt(hidden)).
  float embedding_scale = 1.0f;
  /// \brief First global row of this rank's embedding shard. The token-id
  ///        gather masks ids outside [offset, offset + shard) to zero; the
  ///        embedding all-reduce then completes the lookup. Zero when the
  ///        vocabulary is not sharded.
  int64_t embedding_row_offset = 0;
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
  Tensor token_ids;                  ///< [num_tokens] int32, unless embeddings is set.
  std::optional<Tensor> embeddings;  ///< [num_tokens, hidden] prepared embeddings.
  AttentionBatch attention;          ///< Ragged batch and KV geometry.
};

}  // namespace inferx::causal

#endif  // INFERX_MODELS_CAUSAL_DECODER_CONFIG_H_
