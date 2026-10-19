#ifndef INFERX_MODELS_QWEN3_QWEN3_DECODER_LAYER_H_
#define INFERX_MODELS_QWEN3_QWEN3_DECODER_LAYER_H_

#include <vector>

#include "inferx/config/parallel_config.h"
#include "inferx/core/device.h"
#include "inferx/core/dtype.h"
#include "inferx/core/status.h"
#include "inferx/core/tensor.h"
#include "inferx/models/layers/rms_norm.h"
#include "inferx/models/model.h"
#include "inferx/ops/attention/spec.h"
#include "inferx/models/qwen3/qwen3_attention.h"
#include "inferx/models/qwen3/qwen3_config.h"
#include "inferx/models/qwen3/qwen3_mlp.h"
#include "inferx/models/weight_loader.h"

namespace inferx {
namespace models {
namespace qwen3 {

/// \brief One Qwen3 decoder layer, execution order written out:
///        (add-)norm -> attention -> add-norm -> MLP.
///
/// `residual` is the running residual stream. The first decoder layer has
/// no incoming residual to fuse, so it applies its input norm plain
/// (`fuse_input_norm == false`) and the buffer arrives holding the layer's
/// input; every later layer fuses the add. In both cases the post-attention
/// add-norm updates `residual` in place, and the MLP result is returned
/// un-added -- the next layer's input norm performs that add, exactly one
/// fused add-norm per residual contribution.
class Qwen3DecoderLayer {
 public:
  Qwen3DecoderLayer() = default;

  static StatusOr<Qwen3DecoderLayer> Create(const Qwen3Config& config,
                                            const ParallelConfig& parallel, DataType dtype,
                                            Device device);

  /// \brief Runs the layer.
  ///
  /// \param ctx             Step context.
  /// \param hidden          [tokens, hidden] layer input (attention or MLP
  ///                        output of the previous layer).
  /// \param fuse_input_norm False only for the first decoder layer.
  /// \param residual        [tokens, hidden] running residual; updated.
  /// \param out             [tokens, hidden] MLP output, not residual-added.
  Status Forward(ForwardContext& ctx, const Tensor& positions, const Tensor& hidden,
                 bool fuse_input_norm, Tensor& residual, Tensor& out) const;

  /// \brief Binds `prefix.{input,post_attention}_layernorm.weight` and
  ///        everything the attention and MLP blocks declare.
  Status BindWeights(const std::string& prefix, WeightMap& weights);

  /// \brief The attention layer's spec.
  [[nodiscard]] attn::AttentionSpec Spec() const { return self_attn_.Spec(); }

  /// \brief This layer's parameter manifest, tagged Owner::kLayer.
  [[nodiscard]] std::vector<ParamDecl> DescribeParams(const std::string& prefix,
                                                     int64_t layer) const;

 private:
  Qwen3DecoderLayer(RMSNorm input_layernorm, Qwen3Attention self_attn,
                    RMSNorm post_attention_layernorm, Qwen3MLP mlp)
      : input_layernorm_(std::move(input_layernorm)),
        self_attn_(std::move(self_attn)),
        post_attention_layernorm_(std::move(post_attention_layernorm)),
        mlp_(std::move(mlp)) {}

  RMSNorm input_layernorm_;
  Qwen3Attention self_attn_;
  RMSNorm post_attention_layernorm_;
  Qwen3MLP mlp_;
};

}  // namespace qwen3
}  // namespace models
}  // namespace inferx

#endif  // INFERX_MODELS_QWEN3_QWEN3_DECODER_LAYER_H_
