#ifndef INFERX_MODELS_LLAMA_LLAMA_MODEL_H_
#define INFERX_MODELS_LLAMA_LLAMA_MODEL_H_

#include <vector>

#include "inferx/config/parallel_config.h"
#include "inferx/core/device.h"
#include "inferx/core/dtype.h"
#include "inferx/core/status.h"
#include "inferx/core/tensor.h"
#include "inferx/models/layers/rms_norm.h"
#include "inferx/models/llama/llama_config.h"
#include "inferx/models/llama/llama_decoder_layer.h"
#include "inferx/models/model.h"
#include "inferx/models/weight_loader.h"

namespace inferx {
namespace models {
namespace llama {

/// \brief The Llama transformer tower: decoder layers plus final norm.
class LlamaModel {
 public:
  LlamaModel() = default;

  static StatusOr<LlamaModel> Create(const LlamaConfig& config,
                                     const ParallelConfig& parallel, DataType dtype,
                                     Device device);

  Status Forward(ForwardContext& ctx, const Tensor& positions, const Tensor& hidden_states,
                 Tensor& residual, Tensor& out) const;

  Status BindWeights(const std::string& prefix, WeightMap& weights);

 private:
  LlamaModel(std::vector<LlamaDecoderLayer> layers, RMSNorm norm, int64_t hidden_size)
      : layers_(std::move(layers)), norm_(std::move(norm)), hidden_size_(hidden_size) {}

  std::vector<LlamaDecoderLayer> layers_;
  RMSNorm norm_;
  int64_t hidden_size_ = 0;
};

}  // namespace llama
}  // namespace models
}  // namespace inferx

#endif  // INFERX_MODELS_LLAMA_LLAMA_MODEL_H_
