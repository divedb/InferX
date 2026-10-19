#ifndef INFERX_MODELS_LLAMA_LLAMA_DECODER_LAYER_H_
#define INFERX_MODELS_LLAMA_LLAMA_DECODER_LAYER_H_

#include "inferx/config/parallel_config.h"
#include "inferx/core/device.h"
#include "inferx/core/dtype.h"
#include "inferx/core/status.h"
#include "inferx/core/tensor.h"
#include "inferx/models/layers/rms_norm.h"
#include "inferx/models/llama/llama_attention.h"
#include "inferx/models/llama/llama_config.h"
#include "inferx/models/llama/llama_mlp.h"
#include "inferx/models/model.h"
#include "inferx/ops/attention/spec.h"
#include "inferx/models/weight_loader.h"

namespace inferx {
namespace models {
namespace llama {

/// \brief One Llama decoder layer: (add-)norm -> attention -> add-norm ->
///        MLP. Same residual contract as the Qwen3 layer.
class LlamaDecoderLayer {
 public:
  LlamaDecoderLayer() = default;

  static StatusOr<LlamaDecoderLayer> Create(const LlamaConfig& config,
                                            const ParallelConfig& parallel, DataType dtype,
                                            Device device);

  Status Forward(ForwardContext& ctx, const Tensor& positions, const Tensor& hidden,
                 bool fuse_input_norm, Tensor& residual, Tensor& out) const;

  Status BindWeights(const std::string& prefix, WeightMap& weights);

  /// \brief The attention layer's spec.
  [[nodiscard]] attn::AttentionSpec Spec() const { return self_attn_.Spec(); }

 private:
  LlamaDecoderLayer(RMSNorm input_layernorm, LlamaAttention self_attn,
                    RMSNorm post_attention_layernorm, LlamaMLP mlp)
      : input_layernorm_(std::move(input_layernorm)),
        self_attn_(std::move(self_attn)),
        post_attention_layernorm_(std::move(post_attention_layernorm)),
        mlp_(std::move(mlp)) {}

  RMSNorm input_layernorm_;
  LlamaAttention self_attn_;
  RMSNorm post_attention_layernorm_;
  LlamaMLP mlp_;
};

}  // namespace llama
}  // namespace models
}  // namespace inferx

#endif  // INFERX_MODELS_LLAMA_LLAMA_DECODER_LAYER_H_
