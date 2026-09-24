#include "inferx/models/causal/weight_mapping.h"
#include <cstdlib>
#include <iterator>
#include <optional>
#include <string_view>
#include <variant>
#include <vector>

#include "inferx/engine/parallel_config.h"
#include "inferx/models/checkpoint.h"
#include "inferx/models/components/qkv_linear.h"

namespace inferx::causal {
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
  int64_t offset = 0;
  for (auto* w : weights) {
    INFERX_ASSIGN_OR_RETURN(auto view, packed.Slice(offset, offset + w->Dim(0)));
    INFERX_RETURN_IF_ERROR(w->CopyTo(view));
    offset += w->Dim(0);
    *w = std::move(view);
  }
  return packed;
}

StatusOr<Tensor> Weight(const models::Checkpoint& checkpoint, const std::string& name,
                        std::vector<int64_t> shape, DeviceId device) {
  return checkpoint.UploadBf16(name, Shape(absl::MakeConstSpan(shape)), device);
}

StatusOr<components::LinearWeights> Linear(const models::Checkpoint& checkpoint,
                                       const std::string& prefix,
                                       int64_t out, int64_t in, bool bias, DeviceId device) {
  INFERX_ASSIGN_OR_RETURN(auto weight, Weight(checkpoint, prefix + ".weight", {out, in}, device));
  std::optional<Tensor> bias_weight;
  if (bias) {
    INFERX_ASSIGN_OR_RETURN(bias_weight, Weight(checkpoint, prefix + ".bias", {out}, device));
  }
  return components::LinearWeights{std::move(weight), std::move(bias_weight)};
}

/// \brief Loads the fused, head-sharded QKV projection (QKVParallelLinear).
///
/// Allocates one packed device tensor of [query_rows | kv_rows | kv_rows]
/// rows and uploads this rank's row slices of the checkpoint's q/k/v
/// projections straight into it -- no full-projection intermediates, and no
/// separate device copy pass. The returned per-projection weights are views
/// into the packed allocation, so both stay valid and share storage.
///
/// The config here carries TOTAL head counts; ShardQkv derives this rank's
/// rows and offsets. o_proj stays unsharded: column-parallel loading arrives
/// with the tensor-parallel milestone. Projection biases are never loaded --
/// ValidateExecutable() rejects biased projections before any weight is read.
StatusOr<components::AttentionWeights> LoadAttentionWeights(
    const models::Checkpoint& checkpoint, const std::string& ap,
    const components::AttentionConfig& a, const ParallelConfig& parallel,
    int64_t hidden, DeviceId device) {
  INFERX_ASSIGN_OR_RETURN(auto geometry, components::ShardQkv(a, parallel));
  const int64_t gate_rows = a.output_gate == components::OutputGate::kNone ? 1 : 2;
  const int64_t total_query_rows = a.query_heads * a.head_dim * gate_rows;
  const int64_t total_kv_rows = a.kv_heads * a.head_dim;

  const int64_t packed_rows = geometry.query_rows + 2 * geometry.kv_rows;
  INFERX_ASSIGN_OR_RETURN(
      auto packed, Tensor::Empty(DataType::kBFloat16, Shape({packed_rows, hidden}), device));

  const int64_t query_begin = parallel.tensor_parallel_rank * geometry.query_rows;
  const int64_t kv_begin = geometry.kv_shard * geometry.kv_rows;
  struct Part {
    const char* checkpoint_name;
    int64_t full_rows;
    int64_t begin;
    int64_t rows;
  };
  const Part parts[] = {
      {"q_proj", total_query_rows, query_begin, geometry.query_rows},
      {"k_proj", total_kv_rows, kv_begin, geometry.kv_rows},
      {"v_proj", total_kv_rows, kv_begin, geometry.kv_rows},
  };

  std::optional<Tensor> query_view, key_view, value_view;
  std::optional<Tensor>* views[3] = {&query_view, &key_view, &value_view};
  int64_t offset = 0;
  for (size_t i = 0; i < std::size(parts); ++i) {
    const auto& part = parts[i];
    INFERX_ASSIGN_OR_RETURN(auto host, checkpoint.FindHostBf16(
        ap + part.checkpoint_name + ".weight", Shape({part.full_rows, hidden})));
    INFERX_ASSIGN_OR_RETURN(auto shard, host.Slice(part.begin, part.begin + part.rows));
    INFERX_ASSIGN_OR_RETURN(auto dst, packed.Slice(offset, offset + part.rows));
    INFERX_RETURN_IF_ERROR(shard.CopyTo(dst));
    *views[i] = std::move(dst);
    offset += part.rows;
  }
  const int64_t qdim = a.query_heads * a.head_dim;
  INFERX_ASSIGN_OR_RETURN(auto output, Linear(checkpoint, ap + "o_proj", hidden, qdim,
                                              a.projection_bias, device));
  std::optional<Tensor> query_norm, key_norm;
  if (a.qk_norm) {
    INFERX_ASSIGN_OR_RETURN(query_norm, Weight(checkpoint, ap + "q_norm.weight", {a.head_dim}, device));
    INFERX_ASSIGN_OR_RETURN(key_norm, Weight(checkpoint, ap + "k_norm.weight", {a.head_dim}, device));
  }
  return components::AttentionWeights{
      std::move(packed),
      components::LinearWeights{std::move(*query_view), std::nullopt},
      components::LinearWeights{std::move(*key_view), std::nullopt},
      components::LinearWeights{std::move(*value_view), std::nullopt},
      std::move(output), std::move(query_norm), std::move(key_norm)};
}

StatusOr<components::SwiGluWeights> SwiGlu(const models::Checkpoint& checkpoint,
                                       const std::string& prefix,
                                       int64_t hidden, int64_t width, DeviceId device) {
  INFERX_ASSIGN_OR_RETURN(auto gate, Linear(checkpoint, prefix + "gate_proj", width, hidden, false, device));
  INFERX_ASSIGN_OR_RETURN(auto up, Linear(checkpoint, prefix + "up_proj", width, hidden, false, device));
  INFERX_ASSIGN_OR_RETURN(auto down, Linear(checkpoint, prefix + "down_proj", hidden, width, false, device));
  std::optional<Tensor> packed_gate_up;
  if (PackProjections()) {
    INFERX_ASSIGN_OR_RETURN(packed_gate_up, Pack({&gate.weight, &up.weight}, device));
  }
  return components::SwiGluWeights{std::move(packed_gate_up), std::move(gate), std::move(up),
                               std::move(down)};
}

StatusOr<components::DecoderLayerWeights> LoadDecoderLayer(const models::Checkpoint& checkpoint,
                                         const components::DecoderLayerConfig& config,
                                         const CheckpointLayout& names,
                                         const std::string& prefix, int64_t hidden,
                                         const ParallelConfig& parallel, DeviceId device) {
  INFERX_ASSIGN_OR_RETURN(auto input_norm,
                          Weight(checkpoint, prefix + "input_layernorm.weight", {hidden}, device));
  INFERX_ASSIGN_OR_RETURN(auto post_mixer_norm,
                          Weight(checkpoint, prefix + "post_attention_layernorm.weight", {hidden}, device));
  const auto& a = std::get<components::AttentionConfig>(config.mixer);
  const std::string ap = prefix + names.attention_name;
  INFERX_ASSIGN_OR_RETURN(auto attn, LoadAttentionWeights(checkpoint, ap, a, parallel, hidden, device));
  const std::string fp = prefix + names.feed_forward_name;
  if (const auto* dense = std::get_if<components::SwiGluConfig>(&config.feed_forward)) {
    INFERX_ASSIGN_OR_RETURN(auto ffn, SwiGlu(checkpoint, fp, hidden, dense->intermediate_size, device));
    return components::DecoderLayerWeights{std::move(input_norm), std::move(post_mixer_norm), std::move(attn),
                                std::move(ffn)};
  }
  const auto& m = std::get<components::MoeConfig>(config.feed_forward);
  INFERX_ASSIGN_OR_RETURN(auto router, Weight(checkpoint, fp + "gate.weight", {m.num_experts, hidden}, device));
  std::vector<components::SwiGluWeights> experts;
  for (int64_t i = 0; i < m.num_experts; ++i) {
    INFERX_ASSIGN_OR_RETURN(auto expert, SwiGlu(checkpoint, fp + "experts." + std::to_string(i) + ".", hidden, m.intermediate_size, device));
    experts.push_back(std::move(expert));
  }
  std::optional<components::SwiGluWeights> shared_expert;
  std::optional<Tensor> shared_expert_gate;
  if (m.shared_intermediate_size > 0) {
    INFERX_ASSIGN_OR_RETURN(shared_expert, SwiGlu(checkpoint, fp + "shared_expert.", hidden, m.shared_intermediate_size, device));
    if (m.gate_shared_expert) {
      INFERX_ASSIGN_OR_RETURN(shared_expert_gate, Weight(checkpoint, fp + "shared_expert_gate.weight", {1, hidden}, device));
    }
  }
  components::MoeWeights moe{std::move(router), std::move(experts), std::move(shared_expert),
                         std::move(shared_expert_gate)};
  return components::DecoderLayerWeights{std::move(input_norm), std::move(post_mixer_norm), std::move(attn),
                              std::move(moe)};
}

}  // namespace

StatusOr<Tensor> LoadWeight(const models::Checkpoint& checkpoint, std::string_view name,
                            const Shape& expected, DeviceId device) {
  return checkpoint.UploadBf16(name, expected, device);
}

StatusOr<DecoderWeights> LoadDecoderWeights(const models::Checkpoint& checkpoint,
                                            const DecoderConfig& config,
                                            const CheckpointLayout& names,
                                            const ParallelConfig& parallel, DeviceId device) {
  for (const auto& block : config.blocks) {
    if (!std::holds_alternative<components::AttentionConfig>(block.mixer)) {
      return UnimplementedError("checkpoint mapping for recurrent projections is not implemented");
    }
  }
  const auto& mc = config.model;
  INFERX_ASSIGN_OR_RETURN(auto token_embedding,
      Weight(checkpoint, names.backbone_prefix + "embed_tokens.weight", {mc.vocab_size, mc.hidden_size}, device));
  INFERX_ASSIGN_OR_RETURN(auto final_norm,
      Weight(checkpoint, names.backbone_prefix + "norm.weight", {mc.hidden_size}, device));
  std::vector<components::DecoderLayerWeights> blocks;
  for (size_t i = 0; i < config.blocks.size(); ++i) {
    INFERX_ASSIGN_OR_RETURN(auto block, LoadDecoderLayer(checkpoint, config.blocks[i], names,
        names.backbone_prefix + "layers." + std::to_string(i) + ".", mc.hidden_size, parallel, device));
    blocks.push_back(std::move(block));
  }
  return DecoderWeights{std::move(token_embedding), std::move(final_norm), std::move(blocks)};
}

}  // namespace inferx::causal
