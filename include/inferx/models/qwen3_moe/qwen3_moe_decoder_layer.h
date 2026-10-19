#ifndef INFERX_MODELS_QWEN3_MOE_QWEN3_MOE_DECODER_LAYER_H_
#define INFERX_MODELS_QWEN3_MOE_QWEN3_MOE_DECODER_LAYER_H_

#include "inferx/config/parallel_config.h"
#include "inferx/core/device.h"
#include "inferx/core/dtype.h"
#include "inferx/core/status.h"
#include "inferx/core/tensor.h"
#include "inferx/models/layers/rms_norm.h"
#include "inferx/models/model.h"
#include "inferx/ops/attention/spec.h"
#include "inferx/models/qwen3/qwen3_attention.h"
#include "inferx/models/qwen3_moe/qwen3_moe.h"
#include "inferx/models/qwen3_moe/qwen3_moe_config.h"
#include "inferx/models/weight_loader.h"

namespace inferx {
namespace models {
namespace qwen3_moe {

/// \brief One Qwen3-MoE decoder layer: (add-)norm -> shared attention ->
///        add-norm -> routed experts.
///
/// A separate class from `qwen3::Qwen3DecoderLayer`, not a generalization
/// of it: the feed-forward difference is structural, and each layer reads
/// as its own architecture. Attention is literally the dense family's
/// block, composed by value.
class Qwen3MoeDecoderLayer {
 public:
  Qwen3MoeDecoderLayer() = default;

  static StatusOr<Qwen3MoeDecoderLayer> Create(const Qwen3MoeConfig& config,
                                               const ParallelConfig& parallel,
                                               DataType dtype, Device device);

  /// \see qwen3::Qwen3DecoderLayer::Forward for the residual contract.
  Status Forward(ForwardContext& ctx, const Tensor& positions, const Tensor& hidden,
                 bool fuse_input_norm, Tensor& residual, Tensor& out) const;

  Status BindWeights(const std::string& prefix, WeightMap& weights);

  /// \brief The attention layer's spec.
  [[nodiscard]] attn::AttentionSpec Spec() const { return self_attn_.Spec(); }

 private:
  Qwen3MoeDecoderLayer(RMSNorm input_layernorm, qwen3::Qwen3Attention self_attn,
                       RMSNorm post_attention_layernorm, Qwen3Moe moe)
      : input_layernorm_(std::move(input_layernorm)),
        self_attn_(std::move(self_attn)),
        post_attention_layernorm_(std::move(post_attention_layernorm)),
        moe_(std::move(moe)) {}

  RMSNorm input_layernorm_;
  qwen3::Qwen3Attention self_attn_;
  RMSNorm post_attention_layernorm_;
  Qwen3Moe moe_;
};

}  // namespace qwen3_moe
}  // namespace models
}  // namespace inferx

#endif  // INFERX_MODELS_QWEN3_MOE_QWEN3_MOE_DECODER_LAYER_H_
