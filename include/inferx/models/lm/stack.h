/// \file
/// \brief The generic causal decoder stack: embedding -> pre-norm blocks ->
/// final norm. Phase-agnostic; prefill and decode are expressed entirely in
/// DecoderInput's ragged batch geometry.

#ifndef INFERX_MODELS_LM_STACK_H_
#define INFERX_MODELS_LM_STACK_H_

#include <memory>
#include <vector>

#include "inferx/core/status.h"
#include "inferx/core/tensor.h"
#include "inferx/models/layers/block.h"
#include "inferx/models/model.h"
#include "inferx/models/state.h"
#include "inferx/ops/execution_context.h"
#include "inferx/ops/flash_attention.h"
#include "inferx/ops/linear.h"

namespace inferx::lm {

/// \brief Everything needed to build and validate a decoder stack.
struct DecoderConfig {
  ops::AttentionBackend attention_backend = ops::AttentionBackend::kFlashInfer;
  CheckpointConfig model;                        ///< Family-agnostic dimensions.
  layers::NormConfig final_norm;            ///< Closing norm after the last block.
  std::vector<layers::BlockConfig> blocks;  ///< One entry per layer.

  /// \brief Checks internal consistency of dimensions, geometry, and variants.
  Status Validate() const;

  /// \brief Persistent state each layer needs, one entry per block.
  std::vector<LayerStateSpec> StateRequirements() const;
};

/// \brief All stack weights, on the model's device.
struct DecoderWeights {
  Tensor token_embedding;  ///< [vocab, hidden]
  Tensor final_norm;       ///< [hidden]
  std::vector<layers::BlockWeights> blocks;
};

/// \brief One step's stack input; prepared embeddings may replace token ids.
struct DecoderInput {
  Tensor token_ids;          ///< [num_tokens] int32, unless embeddings is set.
  Tensor embeddings;         ///< [num_tokens, hidden] prepared embeddings.
  AttentionBatch attention;  ///< Ragged batch and KV geometry.
};

/// \brief A token-mixing backbone consumed by task heads.
class Decoder {
 public:
  virtual ~Decoder() = default;

  virtual const CheckpointConfig& config() const = 0;
  virtual std::vector<LayerStateSpec> StateRequirements() const = 0;

  /// \brief Returns [num_tokens, hidden] final hidden states.
  virtual StatusOr<Tensor> Forward(const DecoderInput& input, ModelState& state,
                                   ops::ExecutionContext& ctx) = 0;
};

/// \brief Executes a DecoderConfig over DecoderWeights.
class DecoderStack final : public Decoder {
 public:
  DecoderStack(DecoderConfig config, DecoderWeights weights, int max_tokens);

  const CheckpointConfig& config() const override { return config_.model; }
  std::vector<LayerStateSpec> StateRequirements() const override {
    return config_.StateRequirements();
  }
  StatusOr<Tensor> Forward(const DecoderInput& input, ModelState& state,
                           ops::ExecutionContext& ctx) override;

 private:
  /// \brief Allocates the reusable activation workspace on first use.
  Status InitWorkspace(DeviceId device);

  DecoderConfig config_;
  DecoderWeights weights_;
  int max_tokens_;

  /// \brief Persistent activation workspace, sized by max_tokens_ and the
  /// widest per-block geometry. Forward returns views into these buffers;
  /// they stay valid until the next Forward call.
  bool workspace_ready_ = false;
  int64_t max_intermediate_ = 0;
  bool enable_split_decode_ = false;
  int prefill_tile_rows_ = 64;
  ops::FlashDecodeWorkspace decode_workspace_;
  Tensor attention_plan_;
  Tensor packed_projection_;
  Tensor hidden_, normed_, query_, key_, value_, attn_out_, mixed_, gate_, up_;
};

}  // namespace inferx::lm

#endif  // INFERX_MODELS_LM_STACK_H_
