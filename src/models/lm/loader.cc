#include "inferx/models/lm/loader.h"
#include <cstdlib>
#include <string_view>

#include "inferx/models/lm/causal_lm.h"
#include "models/weight_upload.h"

namespace inferx::lm {
namespace {

bool PackProjections() {
  const char* flag = std::getenv("INFERX_EXPERIMENTAL_PACKED_PROJECTIONS");
  return flag != nullptr && std::string_view(flag) == "1";
}

StatusOr<Tensor> Pack(std::initializer_list<Tensor*> weights, DeviceId device) {
  const int64_t width = (*weights.begin())->Dim(1);
  int64_t rows = 0;
  for (const auto* w : weights) rows += w->Dim(0);
  INFERX_ASSIGN_OR_RETURN(auto packed, Tensor::Empty(DataType::kBFloat16, Shape({rows, width}), device));
  INFERX_ASSIGN_OR_RETURN(auto runtime, RuntimeFor(device));
  int64_t offset = 0;
  for (auto* w : weights) {
    INFERX_ASSIGN_OR_RETURN(auto view, packed.Slice(offset, offset + w->Dim(0)));
    INFERX_RETURN_IF_ERROR(runtime->Copy(view.Data(), w->Data(), w->NBytes(), CopyKind::kDeviceToDevice));
    offset += w->Dim(0);
    *w = std::move(view);
  }
  return packed;
}

StatusOr<Tensor> Weight(const SafeTensors& tensors, const std::string& name,
                        std::vector<int64_t> shape, DeviceId device) {
  const auto* tensor = tensors.Find(name);
  if (tensor == nullptr) return NotFoundError("checkpoint is missing tensor ", name);
  if (tensor->shape != shape) return InvalidArgumentError("unexpected shape for ", name);
  return models::UploadBf16(*tensor, device);
}

StatusOr<layers::LinearWeights> Linear(const SafeTensors& tensors, const std::string& prefix,
                                       int64_t out, int64_t in, bool bias, DeviceId device) {
  layers::LinearWeights weights;
  INFERX_ASSIGN_OR_RETURN(weights.weight, Weight(tensors, prefix + ".weight", {out, in}, device));
  if (bias) {
    INFERX_ASSIGN_OR_RETURN(weights.bias, Weight(tensors, prefix + ".bias", {out}, device));
  }
  return weights;
}

StatusOr<layers::SwiGluWeights> SwiGlu(const SafeTensors& tensors, const std::string& prefix,
                                       int64_t hidden, int64_t width, DeviceId device) {
  layers::SwiGluWeights weights;
  INFERX_ASSIGN_OR_RETURN(weights.gate, Linear(tensors, prefix + "gate_proj", width, hidden, false, device));
  INFERX_ASSIGN_OR_RETURN(weights.up, Linear(tensors, prefix + "up_proj", width, hidden, false, device));
  INFERX_ASSIGN_OR_RETURN(weights.down, Linear(tensors, prefix + "down_proj", hidden, width, false, device));
  if (PackProjections()) {
    INFERX_ASSIGN_OR_RETURN(weights.packed_gate_up, Pack({&weights.gate.weight, &weights.up.weight}, device));
  }
  return weights;
}

StatusOr<layers::BlockWeights> LoadBlock(const SafeTensors& tensors,
                                         const layers::BlockConfig& config,
                                         const CheckpointLayout& names,
                                         const std::string& prefix, int64_t hidden,
                                         DeviceId device) {
  layers::BlockWeights weights;
  INFERX_ASSIGN_OR_RETURN(weights.input_norm, Weight(tensors, prefix + "input_layernorm.weight", {hidden}, device));
  INFERX_ASSIGN_OR_RETURN(weights.post_mixer_norm, Weight(tensors, prefix + "post_attention_layernorm.weight", {hidden}, device));
  const auto& a = std::get<layers::AttentionConfig>(config.mixer);
  const std::string ap = prefix + names.attention_name;
  const int64_t qdim = a.query_heads * a.head_dim;
  const int64_t kvdim = a.kv_heads * a.head_dim;
  layers::AttentionWeights attn;
  INFERX_ASSIGN_OR_RETURN(attn.query, Linear(tensors, ap + "q_proj", qdim * (a.output_gate == layers::OutputGate::kNone ? 1 : 2), hidden, a.projection_bias, device));
  INFERX_ASSIGN_OR_RETURN(attn.key, Linear(tensors, ap + "k_proj", kvdim, hidden, a.projection_bias, device));
  INFERX_ASSIGN_OR_RETURN(attn.value, Linear(tensors, ap + "v_proj", kvdim, hidden, a.projection_bias, device));
  INFERX_ASSIGN_OR_RETURN(attn.output, Linear(tensors, ap + "o_proj", hidden, qdim, a.projection_bias, device));
  if (PackProjections()) {
    INFERX_ASSIGN_OR_RETURN(attn.packed_qkv,
        Pack({&attn.query.weight, &attn.key.weight, &attn.value.weight}, device));
  }
  if (a.qk_norm) {
    INFERX_ASSIGN_OR_RETURN(attn.query_norm, Weight(tensors, ap + "q_norm.weight", {a.head_dim}, device));
    INFERX_ASSIGN_OR_RETURN(attn.key_norm, Weight(tensors, ap + "k_norm.weight", {a.head_dim}, device));
  }
  weights.mixer = std::move(attn);
  const std::string fp = prefix + names.feed_forward_name;
  if (const auto* dense = std::get_if<layers::SwiGluConfig>(&config.feed_forward)) {
    INFERX_ASSIGN_OR_RETURN(auto ffn, SwiGlu(tensors, fp, hidden, dense->intermediate_size, device));
    weights.feed_forward = std::move(ffn);
  } else {
    const auto& m = std::get<layers::MoeConfig>(config.feed_forward);
    layers::MoeWeights moe;
    INFERX_ASSIGN_OR_RETURN(moe.router, Weight(tensors, fp + "gate.weight", {m.num_experts, hidden}, device));
    for (int64_t i = 0; i < m.num_experts; ++i) {
      INFERX_ASSIGN_OR_RETURN(auto expert, SwiGlu(tensors, fp + "experts." + std::to_string(i) + ".", hidden, m.intermediate_size, device));
      moe.experts.push_back(std::move(expert));
    }
    if (m.shared_intermediate_size > 0) {
      INFERX_ASSIGN_OR_RETURN(auto shared, SwiGlu(tensors, fp + "shared_expert.", hidden, m.shared_intermediate_size, device));
      moe.shared_expert = std::move(shared);
      if (m.gate_shared_expert) {
        INFERX_ASSIGN_OR_RETURN(moe.shared_expert_gate, Weight(tensors, fp + "shared_expert_gate.weight", {1, hidden}, device));
      }
    }
    weights.feed_forward = std::move(moe);
  }
  return weights;
}

}  // namespace

StatusOr<std::unique_ptr<Model>> LoadCausalLM(const std::string& directory,
                                              const DecoderConfig& config,
                                              const CheckpointLayout& names,
                                              DeviceId device, int max_tokens,
                                              int max_seqs) {
  INFERX_RETURN_IF_ERROR(config.Validate());
  if (max_tokens <= 0 || max_seqs <= 0) return InvalidArgumentError("model capacities must be positive");
  for (const auto& block : config.blocks) {
    if (!std::holds_alternative<layers::AttentionConfig>(block.mixer)) {
      return UnimplementedError("checkpoint mapping for recurrent projections is not implemented");
    }
  }
  for (const auto& block : config.blocks) {
    const auto& a = std::get<layers::AttentionConfig>(block.mixer);
    ops::AttentionParams p{a.query_heads, a.kv_heads, a.head_dim, 1.0f, a.sliding_window};
    INFERX_RETURN_IF_ERROR(ops::ValidateAttentionGeometry(config.attention_backend, p));
  }
  INFERX_ASSIGN_OR_RETURN(auto tensors, SafeTensors::FromDirectory(directory));
  DecoderWeights weights;
  const auto& mc = config.model;
  INFERX_ASSIGN_OR_RETURN(weights.token_embedding, Weight(tensors, names.backbone_prefix + "embed_tokens.weight", {mc.vocab_size, mc.hidden_size}, device));
  INFERX_ASSIGN_OR_RETURN(weights.final_norm, Weight(tensors, names.backbone_prefix + "norm.weight", {mc.hidden_size}, device));
  for (size_t i = 0; i < config.blocks.size(); ++i) {
    INFERX_ASSIGN_OR_RETURN(auto block, LoadBlock(tensors, config.blocks[i], names,
        names.backbone_prefix + "layers." + std::to_string(i) + ".", mc.hidden_size, device));
    weights.blocks.push_back(std::move(block));
  }
  LanguageModelHead head;
  if (mc.tie_word_embeddings) {
    head.weight = weights.token_embedding;
  } else {
    INFERX_ASSIGN_OR_RETURN(head.weight, Weight(tensors, names.head_name, {mc.vocab_size, mc.hidden_size}, device));
  }
  auto decoder = std::make_unique<DecoderStack>(config, std::move(weights), max_tokens);
  return std::unique_ptr<Model>(new CausalLM(std::move(decoder), std::move(head), max_seqs));
}

}  // namespace inferx::lm
