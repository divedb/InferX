#ifndef INFERX_MODELS_QWEN3_MOE_QWEN3_MOE_MODEL_H_
#define INFERX_MODELS_QWEN3_MOE_QWEN3_MOE_MODEL_H_

#include <vector>

#include "inferx/config/parallel_config.h"
#include "inferx/core/device.h"
#include "inferx/core/dtype.h"
#include "inferx/core/status.h"
#include "inferx/core/tensor.h"
#include "inferx/models/layers/rms_norm.h"
#include "inferx/models/model.h"
#include "inferx/ops/attention/spec.h"
#include "inferx/models/qwen3_moe/qwen3_moe_config.h"
#include "inferx/models/qwen3_moe/qwen3_moe_decoder_layer.h"
#include "inferx/models/weight_loader.h"

namespace inferx {
namespace models {
namespace qwen3_moe {

/// \brief The Qwen3-MoE transformer tower: MoE decoder layers plus final
///        norm. Same residual contract as the dense tower.
class Qwen3MoeModel {
 public:
  Qwen3MoeModel() = default;

  static StatusOr<Qwen3MoeModel> Create(const Qwen3MoeConfig& config,
                                        const ParallelConfig& parallel, DataType dtype,
                                        Device device);

  Status Forward(ForwardContext& ctx, const Tensor& positions, const Tensor& hidden_states,
                 Tensor& residual, Tensor& out) const;

  Status BindWeights(const std::string& prefix, WeightMap& weights);

  const std::vector<Qwen3MoeDecoderLayer>& Layers() const { return layers_; }

  /// \brief One attention spec per decoder layer, in tower order.
  [[nodiscard]] std::vector<attn::AttentionSpec> AttentionSpecs() const;

 private:
  Qwen3MoeModel(std::vector<Qwen3MoeDecoderLayer> layers, RMSNorm norm,
                int64_t hidden_size)
      : layers_(std::move(layers)), norm_(std::move(norm)), hidden_size_(hidden_size) {}

  std::vector<Qwen3MoeDecoderLayer> layers_;
  RMSNorm norm_;
  int64_t hidden_size_ = 0;
};

}  // namespace qwen3_moe
}  // namespace models
}  // namespace inferx

#endif  // INFERX_MODELS_QWEN3_MOE_QWEN3_MOE_MODEL_H_
