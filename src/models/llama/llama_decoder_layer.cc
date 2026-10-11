#include "inferx/models/llama/llama_decoder_layer.h"

#include <utility>

namespace inferx {
namespace models {
namespace llama {

StatusOr<LlamaDecoderLayer> LlamaDecoderLayer::Create(const LlamaConfig& config,
                                                      const ParallelConfig& parallel,
                                                      DataType dtype, Device device) {
  INFERX_ASSIGN_OR_RETURN(
      RMSNorm input_layernorm,
      RMSNorm::Create(config.hidden_size, config.rms_norm_eps, dtype, device));
  INFERX_ASSIGN_OR_RETURN(LlamaAttention self_attn,
                          LlamaAttention::Create(config, parallel, dtype, device));
  INFERX_ASSIGN_OR_RETURN(
      RMSNorm post_attention_layernorm,
      RMSNorm::Create(config.hidden_size, config.rms_norm_eps, dtype, device));
  INFERX_ASSIGN_OR_RETURN(LlamaMLP mlp, LlamaMLP::Create(config, parallel, dtype, device));
  return LlamaDecoderLayer(std::move(input_layernorm), std::move(self_attn),
                           std::move(post_attention_layernorm), std::move(mlp));
}

Status LlamaDecoderLayer::Forward(ForwardContext& ctx, const Tensor& positions,
                                  const Tensor& hidden, bool fuse_input_norm,
                                  Tensor& residual, Tensor& out) const {
  const int64_t tokens = hidden.Dim(0);
  const DataType dtype = hidden.GetDataType();
  const int64_t hidden_size = hidden.Dim(1);

  INFERX_ASSIGN_OR_RETURN(Tensor attn_in,
                          Tensor::Empty(dtype, Shape({tokens, hidden_size}), &ctx.scratch));
  if (fuse_input_norm) {
    INFERX_RETURN_IF_ERROR(input_layernorm_.AddForward(ctx.op, hidden, residual, attn_in));
  } else {
    INFERX_RETURN_IF_ERROR(input_layernorm_.Forward(ctx.op, hidden, attn_in));
  }

  INFERX_ASSIGN_OR_RETURN(Tensor attn_out,
                          Tensor::Empty(dtype, Shape({tokens, hidden_size}), &ctx.scratch));
  INFERX_RETURN_IF_ERROR(self_attn_.Forward(ctx, positions, attn_in, attn_out));

  INFERX_ASSIGN_OR_RETURN(Tensor mlp_in,
                          Tensor::Empty(dtype, Shape({tokens, hidden_size}), &ctx.scratch));
  INFERX_RETURN_IF_ERROR(
      post_attention_layernorm_.AddForward(ctx.op, attn_out, residual, mlp_in));

  return mlp_.Forward(ctx, mlp_in, out);
}

Status LlamaDecoderLayer::BindWeights(const std::string& prefix, WeightMap& weights) {
  INFERX_RETURN_IF_ERROR(
      weights.Add(prefix + ".input_layernorm.weight",
                  WeightBinding::Replicate(input_layernorm_.MutableWeight())));
  INFERX_RETURN_IF_ERROR(self_attn_.BindWeights(prefix + ".self_attn", weights));
  INFERX_RETURN_IF_ERROR(
      weights.Add(prefix + ".post_attention_layernorm.weight",
                  WeightBinding::Replicate(post_attention_layernorm_.MutableWeight())));
  return mlp_.BindWeights(prefix + ".mlp", weights);
}

}  // namespace llama
}  // namespace models
}  // namespace inferx
