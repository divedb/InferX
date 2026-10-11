#include "inferx/models/qwen3_moe/qwen3_moe_model.h"

#include <utility>

namespace inferx {
namespace models {
namespace qwen3_moe {

StatusOr<Qwen3MoeModel> Qwen3MoeModel::Create(const Qwen3MoeConfig& config,
                                              const ParallelConfig& parallel,
                                              DataType dtype, Device device) {
  std::vector<Qwen3MoeDecoderLayer> layers;
  layers.reserve(static_cast<size_t>(config.base.num_hidden_layers));
  for (int64_t i = 0; i < config.base.num_hidden_layers; ++i) {
    INFERX_ASSIGN_OR_RETURN(Qwen3MoeDecoderLayer layer,
                            Qwen3MoeDecoderLayer::Create(config, parallel, dtype, device));
    layers.push_back(std::move(layer));
  }
  INFERX_ASSIGN_OR_RETURN(RMSNorm norm,
                          RMSNorm::Create(config.base.hidden_size,
                                          config.base.rms_norm_eps, dtype, device));
  return Qwen3MoeModel(std::move(layers), std::move(norm), config.base.hidden_size);
}

Status Qwen3MoeModel::Forward(ForwardContext& ctx, const Tensor& positions,
                              const Tensor& hidden_states, Tensor& residual,
                              Tensor& out) const {
  const int64_t tokens = hidden_states.Dim(0);
  const DataType dtype = hidden_states.GetDataType();

  Tensor current = hidden_states;
  for (size_t i = 0; i < layers_.size(); ++i) {
    ctx.layer = static_cast<int64_t>(i);
    INFERX_ASSIGN_OR_RETURN(
        Tensor next, Tensor::Empty(dtype, Shape({tokens, hidden_size_}), &ctx.scratch));
    INFERX_RETURN_IF_ERROR(
        layers_[i].Forward(ctx, positions, current, i > 0, residual, next));
    current = std::move(next);
  }
  return norm_.AddForward(ctx.op, current, residual, out);
}

Status Qwen3MoeModel::BindWeights(const std::string& prefix, WeightMap& weights) {
  for (size_t i = 0; i < layers_.size(); ++i) {
    INFERX_RETURN_IF_ERROR(layers_[i].BindWeights(
        prefix + ".layers." + std::to_string(i), weights));
  }
  return weights.Add(prefix + ".norm.weight",
                     WeightBinding::Replicate(norm_.MutableWeight()));
}


}  // namespace qwen3_moe
}  // namespace models
}  // namespace inferx
