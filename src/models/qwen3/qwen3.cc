#include "inferx/models/qwen3/qwen3.h"

#include <string>
#include <utility>

#include "inferx/models/safetensors.h"
#include "models/weight_upload.h"

namespace inferx {
namespace {

using models::GetWeight;

StatusOr<Qwen3LayerWeights> LoadLayer(const SafeTensors& tensors, int layer, DeviceId device) {
  const std::string prefix = "model.layers." + std::to_string(layer) + ".";
  Qwen3LayerWeights weights;
  INFERX_ASSIGN_OR_RETURN(weights.input_layernorm,
                          GetWeight(tensors, prefix + "input_layernorm.weight", device));
  INFERX_ASSIGN_OR_RETURN(
      weights.post_attention_layernorm,
      GetWeight(tensors, prefix + "post_attention_layernorm.weight", device));
  INFERX_ASSIGN_OR_RETURN(weights.attn.q_proj,
                          GetWeight(tensors, prefix + "self_attn.q_proj.weight", device));
  INFERX_ASSIGN_OR_RETURN(weights.attn.k_proj,
                          GetWeight(tensors, prefix + "self_attn.k_proj.weight", device));
  INFERX_ASSIGN_OR_RETURN(weights.attn.v_proj,
                          GetWeight(tensors, prefix + "self_attn.v_proj.weight", device));
  INFERX_ASSIGN_OR_RETURN(weights.attn.o_proj,
                          GetWeight(tensors, prefix + "self_attn.o_proj.weight", device));
  INFERX_ASSIGN_OR_RETURN(weights.attn.q_norm,
                          GetWeight(tensors, prefix + "self_attn.q_norm.weight", device));
  INFERX_ASSIGN_OR_RETURN(weights.attn.k_norm,
                          GetWeight(tensors, prefix + "self_attn.k_norm.weight", device));
  INFERX_ASSIGN_OR_RETURN(weights.mlp.gate_proj,
                          GetWeight(tensors, prefix + "mlp.gate_proj.weight", device));
  INFERX_ASSIGN_OR_RETURN(weights.mlp.up_proj,
                          GetWeight(tensors, prefix + "mlp.up_proj.weight", device));
  INFERX_ASSIGN_OR_RETURN(weights.mlp.down_proj,
                          GetWeight(tensors, prefix + "mlp.down_proj.weight", device));
  return weights;
}

}  // namespace

StatusOr<Qwen3Weights> Qwen3Weights::Load(const std::string& model_dir, DeviceId device) {
  INFERX_ASSIGN_OR_RETURN(ModelConfig config,
                          ModelConfig::FromFile(model_dir + "/config.json"));
  if (config.model_type != "qwen3") {
    return InvalidArgumentError("expected model_type qwen3, got ", config.model_type);
  }
  if (config.hidden_act != "silu") {
    return InvalidArgumentError("expected hidden_act silu, got ", config.hidden_act);
  }
  if (config.head_dim % 2 != 0) {
    return InvalidArgumentError("head_dim must be even for RoPE, got ", config.head_dim);
  }

  INFERX_ASSIGN_OR_RETURN(SafeTensors tensors, SafeTensors::FromDirectory(model_dir));

  Qwen3Weights weights;
  weights.config = config;
  INFERX_ASSIGN_OR_RETURN(weights.embed_tokens,
                          GetWeight(tensors, "model.embed_tokens.weight", device));
  INFERX_ASSIGN_OR_RETURN(weights.norm, GetWeight(tensors, "model.norm.weight", device));

  // tie_word_embeddings checkpoints may omit lm_head; the tied model reuses
  // the embedding table.
  if (tensors.Contains("lm_head.weight")) {
    INFERX_ASSIGN_OR_RETURN(weights.lm_head, GetWeight(tensors, "lm_head.weight", device));
  } else if (config.tie_word_embeddings) {
    weights.lm_head = weights.embed_tokens;
  } else {
    return NotFoundError("checkpoint has no lm_head.weight and embeddings are not tied");
  }

  weights.layers.resize(static_cast<size_t>(config.num_hidden_layers));
  for (int i = 0; i < config.num_hidden_layers; ++i) {
    INFERX_ASSIGN_OR_RETURN(weights.layers[static_cast<size_t>(i)],
                            LoadLayer(tensors, i, device));
  }
  return weights;
}

Qwen3Model::Qwen3Model(Qwen3Weights&& weights) : weights_(std::move(weights)) {}

StatusOr<std::unique_ptr<Model>> Qwen3Model::Load(const std::string& directory, DeviceId device,
                                                  int max_tokens, int max_seqs) {
  if (max_tokens <= 0 || max_seqs <= 0) {
    return InvalidArgumentError("model workspace capacities must be positive");
  }
  INFERX_ASSIGN_OR_RETURN(Qwen3Weights weights, Qwen3Weights::Load(directory, device));
  return std::unique_ptr<Model>(new Qwen3Model(std::move(weights)));
}

StatusOr<Tensor> Qwen3Model::Forward(const ModelInput& input, KvBlockPool& /*cache*/,
                                     ops::ExecutionContext& ctx) {
  const int rows = input.attention.num_tokens;
  if (rows <= 0) {
    return InvalidArgumentError("Qwen3 forward requires a non-empty token batch");
  }
  if (!input.token_ids.IsDefined() || input.token_ids.Numel() != rows ||
      input.logit_rows.Numel() != input.attention.num_seqs) {
    return InvalidArgumentError("Qwen3 forward input shapes disagree with batch metadata");
  }
  if (input.token_ids.Device() != ctx.device()) {
    return InvalidArgumentError("Qwen3 forward inputs must live on the context's device");
  }
  // The Qwen3 flow, filled in as ops land:
  //   hidden = gather(embed_tokens, token_ids)
  //   per layer: fused-add RMSNorm(input_layernorm)            [ops::RmsNorm]
  //              q/k/v = Linear(hidden, {q,k,v}_proj)
  //              q/k = RmsNorm(q/k, q_norm/k_norm, head_dim)   [ops::RmsNorm]
  //              q/k = RoPE(q/k, positions)
  //              attn_out = Attention(cache, layer, q, k, v)
  //              hidden += Linear(attn_out, o_proj)
  //              fused-add RMSNorm(post_attention_layernorm)
  //              hidden += Linear(GatedSilu(hidden, mlp), down_proj)
  //   fused-add RMSNorm(norm); logits = lm_head(hidden[logit_rows])
  return UnimplementedError(
      "Qwen3 forward awaits ops: embedding gather, linear, fused-add rmsnorm, rope, "
      "attention, gated-silu");
}

}  // namespace inferx
