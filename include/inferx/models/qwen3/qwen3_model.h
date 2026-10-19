#ifndef INFERX_MODELS_QWEN3_QWEN3_MODEL_H_
#define INFERX_MODELS_QWEN3_QWEN3_MODEL_H_

#include <vector>

#include "inferx/config/parallel_config.h"
#include "inferx/core/device.h"
#include "inferx/core/dtype.h"
#include "inferx/core/status.h"
#include "inferx/core/tensor.h"
#include "inferx/models/layers/rms_norm.h"
#include "inferx/models/model.h"
#include "inferx/models/qwen3/qwen3_config.h"
#include "inferx/models/qwen3/qwen3_decoder_layer.h"

namespace inferx {
namespace models {
namespace qwen3 {

/// \brief The Qwen3 transformer tower: decoder layers plus final norm.
///
/// Embedding and LM head belong to `Qwen3ForCausalLM`; this class is the
/// stack that maps hidden states to hidden states, so dense and MoE
/// variants differ only in which decoder layer they hold.
class Qwen3Model {
 public:
  Qwen3Model() = default;

  static StatusOr<Qwen3Model> Create(const Qwen3Config& config,
                                     const ParallelConfig& parallel, DataType dtype,
                                     Device device);

  /// \brief Runs the stack over pre-embedded hidden states.
  ///
  /// `residual` arrives holding the embedding output and leaves holding the
  /// full residual stream after the last layer; `out` receives the
  /// final-norm activations for every token.
  Status Forward(ForwardContext& ctx, const Tensor& positions, const Tensor& hidden_states,
                 Tensor& residual, Tensor& out) const;

  Status BindWeights(const std::string& prefix, WeightMap& weights);

  const std::vector<Qwen3DecoderLayer>& Layers() const { return layers_; }

  /// \brief One attention spec per decoder layer, in tower order.
  [[nodiscard]] std::vector<attn::AttentionSpec> AttentionSpecs() const;

 private:
  Qwen3Model(std::vector<Qwen3DecoderLayer> layers, RMSNorm norm)
      : layers_(std::move(layers)), norm_(std::move(norm)) {}

  std::vector<Qwen3DecoderLayer> layers_;
  RMSNorm norm_;
};

}  // namespace qwen3
}  // namespace models
}  // namespace inferx

#endif  // INFERX_MODELS_QWEN3_QWEN3_MODEL_H_
