#include "inferx/models/qwen3/qwen3_model.h"

#include <utility>

namespace inferx {
namespace models {
namespace qwen3 {

StatusOr<Qwen3Model> Qwen3Model::Create(const Qwen3Config& config,
                                        const ParallelConfig& parallel, DataType dtype,
                                        Device device) {
  std::vector<Qwen3DecoderLayer> layers;
  layers.reserve(static_cast<size_t>(config.num_hidden_layers));
  for (int64_t i = 0; i < config.num_hidden_layers; ++i) {
    INFERX_ASSIGN_OR_RETURN(Qwen3DecoderLayer layer,
                            Qwen3DecoderLayer::Create(config, parallel, dtype, device));
    layers.push_back(std::move(layer));
  }
  INFERX_ASSIGN_OR_RETURN(RMSNorm norm,
                          RMSNorm::Create(config.hidden_size, config.rms_norm_eps, dtype,
                                          device));
  return Qwen3Model(std::move(layers), std::move(norm));
}

Status Qwen3Model::Forward(ForwardContext& ctx, const Tensor& positions,
                           const Tensor& hidden_states, Tensor& residual,
                           Tensor& out) const {
  // Ping-pong the layer input/output through two scratch buffers; the
  // residual stream owns its own buffer for the whole stack.
  const int64_t tokens = hidden_states.Dim(0);
  const DataType dtype = hidden_states.GetDataType();
  const int64_t hidden = hidden_states.Dim(1);

  Tensor current = hidden_states;
  for (size_t i = 0; i < layers_.size(); ++i) {
    ctx.layer = static_cast<int64_t>(i);
    INFERX_ASSIGN_OR_RETURN(Tensor next,
                            Tensor::Empty(dtype, Shape({tokens, hidden}), &ctx.scratch));
    INFERX_RETURN_IF_ERROR(
        layers_[i].Forward(ctx, positions, current, i > 0, residual, next));
    current = std::move(next);
  }

  // The last MLP output is still un-added; the final norm performs that add.
  return norm_.AddForward(ctx.op, current, residual, out);
}

Status Qwen3Model::BindWeights(const std::string& prefix, WeightMap& weights) {
  for (size_t i = 0; i < layers_.size(); ++i) {
    INFERX_RETURN_IF_ERROR(layers_[i].BindWeights(
        prefix + ".layers." + std::to_string(i), weights));
  }
  return weights.Add(prefix + ".norm.weight",
                     WeightBinding::Replicate(norm_.MutableWeight()));
}


}  // namespace qwen3
}  // namespace models
}  // namespace inferx
