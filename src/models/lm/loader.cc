#include "inferx/models/lm/loader.h"
#include <cstdlib>
#include <optional>
#include <string_view>
#include <variant>
#include <vector>

#include "inferx/models/checkpoint.h"
#include "inferx/models/lm/causal_lm.h"

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

StatusOr<Tensor> Weight(const models::Checkpoint& checkpoint, const std::string& name,
                        std::vector<int64_t> shape, DeviceId device) {
  return checkpoint.UploadBf16(name, Shape(absl::MakeConstSpan(shape)), device);
}

StatusOr<layers::LinearWeights> Linear(const models::Checkpoint& checkpoint,
                                       const std::string& prefix,
                                       int64_t out, int64_t in, bool bias, DeviceId device) {
  INFERX_ASSIGN_OR_RETURN(auto weight, Weight(checkpoint, prefix + ".weight", {out, in}, device));
  std::optional<Tensor> bias_weight;
  if (bias) {
    INFERX_ASSIGN_OR_RETURN(bias_weight, Weight(checkpoint, prefix + ".bias", {out}, device));
  }
  return layers::LinearWeights{std::move(weight), std::move(bias_weight)};
}

StatusOr<layers::SwiGluWeights> SwiGlu(const models::Checkpoint& checkpoint,
                                       const std::string& prefix,
                                       int64_t hidden, int64_t width, DeviceId device) {
  INFERX_ASSIGN_OR_RETURN(auto gate, Linear(checkpoint, prefix + "gate_proj", width, hidden, false, device));
  INFERX_ASSIGN_OR_RETURN(auto up, Linear(checkpoint, prefix + "up_proj", width, hidden, false, device));
  INFERX_ASSIGN_OR_RETURN(auto down, Linear(checkpoint, prefix + "down_proj", hidden, width, false, device));
  std::optional<Tensor> packed_gate_up;
  if (PackProjections()) {
    INFERX_ASSIGN_OR_RETURN(packed_gate_up, Pack({&gate.weight, &up.weight}, device));
  }
  return layers::SwiGluWeights{std::move(packed_gate_up), std::move(gate), std::move(up),
                               std::move(down)};
}

StatusOr<layers::BlockWeights> LoadBlock(const models::Checkpoint& checkpoint,
                                         const layers::BlockConfig& config,
                                         const CheckpointLayout& names,
                                         const std::string& prefix, int64_t hidden,
                                         DeviceId device) {
  INFERX_ASSIGN_OR_RETURN(auto input_norm,
                          Weight(checkpoint, prefix + "input_layernorm.weight", {hidden}, device));
  INFERX_ASSIGN_OR_RETURN(auto post_mixer_norm,
                          Weight(checkpoint, prefix + "post_attention_layernorm.weight", {hidden}, device));
  const auto& a = std::get<layers::AttentionConfig>(config.mixer);
  const std::string ap = prefix + names.attention_name;
  const int64_t qdim = a.query_heads * a.head_dim;
  const int64_t kvdim = a.kv_heads * a.head_dim;
  INFERX_ASSIGN_OR_RETURN(auto query, Linear(checkpoint, ap + "q_proj", qdim * (a.output_gate == layers::OutputGate::kNone ? 1 : 2), hidden, a.projection_bias, device));
  INFERX_ASSIGN_OR_RETURN(auto key, Linear(checkpoint, ap + "k_proj", kvdim, hidden, a.projection_bias, device));
  INFERX_ASSIGN_OR_RETURN(auto value, Linear(checkpoint, ap + "v_proj", kvdim, hidden, a.projection_bias, device));
  INFERX_ASSIGN_OR_RETURN(auto output, Linear(checkpoint, ap + "o_proj", hidden, qdim, a.projection_bias, device));
  std::optional<Tensor> packed_qkv;
  if (PackProjections()) {
    INFERX_ASSIGN_OR_RETURN(packed_qkv,
        Pack({&query.weight, &key.weight, &value.weight}, device));
  }
  std::optional<Tensor> query_norm, key_norm;
  if (a.qk_norm) {
    INFERX_ASSIGN_OR_RETURN(query_norm, Weight(checkpoint, ap + "q_norm.weight", {a.head_dim}, device));
    INFERX_ASSIGN_OR_RETURN(key_norm, Weight(checkpoint, ap + "k_norm.weight", {a.head_dim}, device));
  }
  layers::AttentionWeights attn{std::move(packed_qkv), std::move(query), std::move(key),
                                std::move(value), std::move(output), std::move(query_norm),
                                std::move(key_norm)};
  const std::string fp = prefix + names.feed_forward_name;
  if (const auto* dense = std::get_if<layers::SwiGluConfig>(&config.feed_forward)) {
    INFERX_ASSIGN_OR_RETURN(auto ffn, SwiGlu(checkpoint, fp, hidden, dense->intermediate_size, device));
    return layers::BlockWeights{std::move(input_norm), std::move(post_mixer_norm), std::move(attn),
                                std::move(ffn)};
  }
  const auto& m = std::get<layers::MoeConfig>(config.feed_forward);
  INFERX_ASSIGN_OR_RETURN(auto router, Weight(checkpoint, fp + "gate.weight", {m.num_experts, hidden}, device));
  std::vector<layers::SwiGluWeights> experts;
  for (int64_t i = 0; i < m.num_experts; ++i) {
    INFERX_ASSIGN_OR_RETURN(auto expert, SwiGlu(checkpoint, fp + "experts." + std::to_string(i) + ".", hidden, m.intermediate_size, device));
    experts.push_back(std::move(expert));
  }
  std::optional<layers::SwiGluWeights> shared_expert;
  std::optional<Tensor> shared_expert_gate;
  if (m.shared_intermediate_size > 0) {
    INFERX_ASSIGN_OR_RETURN(shared_expert, SwiGlu(checkpoint, fp + "shared_expert.", hidden, m.shared_intermediate_size, device));
    if (m.gate_shared_expert) {
      INFERX_ASSIGN_OR_RETURN(shared_expert_gate, Weight(checkpoint, fp + "shared_expert_gate.weight", {1, hidden}, device));
    }
  }
  layers::MoeWeights moe{std::move(router), std::move(experts), std::move(shared_expert),
                         std::move(shared_expert_gate)};
  return layers::BlockWeights{std::move(input_norm), std::move(post_mixer_norm), std::move(attn),
                              std::move(moe)};
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
  INFERX_ASSIGN_OR_RETURN(auto checkpoint, models::Checkpoint::Open(directory));
  const auto& mc = config.model;
  INFERX_ASSIGN_OR_RETURN(auto token_embedding,
      Weight(checkpoint, names.backbone_prefix + "embed_tokens.weight", {mc.vocab_size, mc.hidden_size}, device));
  INFERX_ASSIGN_OR_RETURN(auto final_norm,
      Weight(checkpoint, names.backbone_prefix + "norm.weight", {mc.hidden_size}, device));
  std::vector<layers::BlockWeights> blocks;
  for (size_t i = 0; i < config.blocks.size(); ++i) {
    INFERX_ASSIGN_OR_RETURN(auto block, LoadBlock(checkpoint, config.blocks[i], names,
        names.backbone_prefix + "layers." + std::to_string(i) + ".", mc.hidden_size, device));
    blocks.push_back(std::move(block));
  }
  DecoderWeights weights{std::move(token_embedding), std::move(final_norm), std::move(blocks)};
  std::optional<Tensor> head_weight;
  if (mc.tie_word_embeddings) {
    head_weight = weights.token_embedding;
  } else {
    INFERX_ASSIGN_OR_RETURN(head_weight, Weight(checkpoint, names.head_name, {mc.vocab_size, mc.hidden_size}, device));
  }
  LanguageModelHead head{std::move(*head_weight)};
  auto decoder = std::make_unique<DecoderStack>(config, std::move(weights), max_tokens);
  return std::unique_ptr<Model>(new CausalLM(std::move(decoder), std::move(head), max_seqs));
}

}  // namespace inferx::lm
