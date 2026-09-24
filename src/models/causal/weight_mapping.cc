#include "inferx/models/causal/weight_mapping.h"
#include <cstring>
#include <iterator>
#include <optional>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include "inferx/engine/parallel_config.h"
#include "inferx/models/checkpoint.h"
#include "inferx/models/components/parallel_linear.h"
#include "inferx/models/components/qkv_linear.h"

namespace inferx::causal {
namespace {

StatusOr<Tensor> Weight(const models::Checkpoint& checkpoint, const std::string& name,
                        std::vector<int64_t> shape, DeviceId device) {
  return checkpoint.UploadBf16(name, Shape(absl::MakeConstSpan(shape)), device);
}

/// \brief Copies columns [begin, end) of a rank-2 host tensor into a fresh
///        contiguous host tensor.
///
/// RowParallelLinear shards along the input dimension, which dimension-0
/// Tensor slices cannot express; this gather runs on the host before
/// upload. Only tensor-parallel loads pay it -- a single rank uploads the
/// checkpoint tensor directly.
StatusOr<Tensor> CopyColumnRange(const Tensor& src, int64_t begin, int64_t end) {
  if (src.Rank() != 2 || begin < 0 || end <= begin || end > src.Dim(1)) {
    return InvalidArgumentError("invalid column range [", begin, ", ", end, ") of ",
                                src.GetShape().ToString());
  }
  const int64_t rows = src.Dim(0);
  const int64_t width = end - begin;
  const int64_t elem = DataTypeByteSize(src.GetDataType(), 1);
  INFERX_ASSIGN_OR_RETURN(
      auto dst, Tensor::Empty(src.GetDataType(), Shape({rows, width}), DeviceId::Cpu()));
  for (int64_t r = 0; r < rows; ++r) {
    const auto* from = src.Bytes() + (r * src.Dim(1) + begin) * elem;
    auto* to = dst.Bytes() + r * width * elem;
    std::memmove(to, from, static_cast<size_t>(width * elem));
  }
  return dst;
}

/// \brief Loads a RowParallelLinear weight: full tensor is [rows, total_cols],
///        this rank gets its input-column shard [rows, shard.size].
///
/// The rank-local GEMM yields partial sums over its input slice; the
/// cross-rank reduction arrives with the tensor-parallel milestone.
StatusOr<Tensor> LoadRowParallel(const models::Checkpoint& checkpoint, const std::string& name,
                                 int64_t rows, int64_t total_cols,
                                 const ParallelConfig& parallel, DeviceId device) {
  INFERX_ASSIGN_OR_RETURN(auto shard, components::ShardDim(total_cols, parallel));
  if (parallel.tensor_parallel_size == 1) {
    return checkpoint.UploadBf16(name, Shape({rows, total_cols}), device);
  }
  INFERX_ASSIGN_OR_RETURN(auto host, checkpoint.FindHostBf16(name, Shape({rows, total_cols})));
  INFERX_ASSIGN_OR_RETURN(auto gathered,
                          CopyColumnRange(host, shard.begin, shard.begin + shard.size));
  return gathered.To(device);
}

/// \brief Loads the fused, head-agnostic QKV projection (QKVParallelLinear).
///
/// Allocates one packed device tensor of [query_rows | kv_rows | kv_rows]
/// rows and uploads this rank's row slices of the checkpoint's q/k/v
/// projections straight into it -- no full-projection intermediates, and no
/// separate device copy pass. The returned per-projection weights are views
/// into the packed allocation, so both stay valid and share storage.
///
/// The config here carries TOTAL head counts; ShardQkv derives this rank's
/// rows and offsets. o_proj is RowParallel: its input columns shard by this
/// rank's (undoubled) query width. Projection biases are never loaded --
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
  const int64_t total_query_width = a.query_heads * a.head_dim;  // Undoubled: o_proj input.
  INFERX_ASSIGN_OR_RETURN(auto output,
                          LoadRowParallel(checkpoint, ap + "o_proj.weight", hidden,
                                          total_query_width, parallel, device));
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
      components::LinearWeights{std::move(output), std::nullopt},
      std::move(query_norm), std::move(key_norm)};
}

/// \brief Loads a SwiGLU block: fused gate|up (MergedColumnParallelLinear --
///        one [2 * shard, hidden] allocation, per-rank row slices uploaded
///        directly) plus a RowParallel down projection.
StatusOr<components::SwiGluWeights> SwiGlu(const models::Checkpoint& checkpoint,
                                       const std::string& prefix, int64_t hidden,
                                       int64_t total_intermediate,
                                       const ParallelConfig& parallel, DeviceId device) {
  INFERX_ASSIGN_OR_RETURN(auto shard, components::ShardDim(total_intermediate, parallel));
  const int64_t rows = shard.size;
  INFERX_ASSIGN_OR_RETURN(auto packed,
      Tensor::Empty(DataType::kBFloat16, Shape({2 * rows, hidden}), device));
  const struct { const char* name; int64_t offset; } parts[] = {
      {"gate_proj", 0}, {"up_proj", rows}};
  for (const auto& part : parts) {
    INFERX_ASSIGN_OR_RETURN(auto host, checkpoint.FindHostBf16(
        prefix + part.name + ".weight", Shape({total_intermediate, hidden})));
    INFERX_ASSIGN_OR_RETURN(auto slice, host.Slice(shard.begin, shard.begin + rows));
    INFERX_ASSIGN_OR_RETURN(auto dst, packed.Slice(part.offset, part.offset + rows));
    INFERX_RETURN_IF_ERROR(slice.CopyTo(dst));
  }
  INFERX_ASSIGN_OR_RETURN(Tensor gate_view, packed.Slice(0, rows));
  INFERX_ASSIGN_OR_RETURN(Tensor up_view, packed.Slice(rows, 2 * rows));
  INFERX_ASSIGN_OR_RETURN(auto down,
                          LoadRowParallel(checkpoint, prefix + "down_proj.weight", hidden,
                                          total_intermediate, parallel, device));
  return components::SwiGluWeights{std::move(packed),
      components::LinearWeights{std::move(gate_view), std::nullopt},
      components::LinearWeights{std::move(up_view), std::nullopt},
      components::LinearWeights{std::move(down), std::nullopt}};
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
    INFERX_ASSIGN_OR_RETURN(auto ffn, SwiGlu(checkpoint, fp, hidden, dense->intermediate_size,
                                             parallel, device));
    return components::DecoderLayerWeights{std::move(input_norm), std::move(post_mixer_norm), std::move(attn),
                                std::move(ffn)};
  }
  const auto& m = std::get<components::MoeConfig>(config.feed_forward);
  INFERX_ASSIGN_OR_RETURN(auto router, Weight(checkpoint, fp + "gate.weight", {m.num_experts, hidden}, device));
  std::vector<components::SwiGluWeights> experts;
  for (int64_t i = 0; i < m.num_experts; ++i) {
    INFERX_ASSIGN_OR_RETURN(auto expert, SwiGlu(checkpoint, fp + "experts." + std::to_string(i) + ".", hidden,
                                                m.intermediate_size, parallel, device));
    experts.push_back(std::move(expert));
  }
  std::optional<components::SwiGluWeights> shared_expert;
  std::optional<Tensor> shared_expert_gate;
  if (m.shared_intermediate_size > 0) {
    INFERX_ASSIGN_OR_RETURN(shared_expert, SwiGlu(checkpoint, fp + "shared_expert.", hidden,
                                                  m.shared_intermediate_size, parallel, device));
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

StatusOr<Tensor> LoadVocabShard(const models::Checkpoint& checkpoint, std::string_view name,
                                int64_t vocab, int64_t hidden, const ParallelConfig& parallel,
                                DeviceId device) {
  INFERX_ASSIGN_OR_RETURN(auto shard, components::ShardDim(vocab, parallel));
  INFERX_ASSIGN_OR_RETURN(auto host,
                          checkpoint.FindHostBf16(name, Shape({vocab, hidden})));
  INFERX_ASSIGN_OR_RETURN(auto slice, host.Slice(shard.begin, shard.begin + shard.size));
  return slice.To(device);
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
      LoadVocabShard(checkpoint, names.backbone_prefix + "embed_tokens.weight", mc.vocab_size,
                     mc.hidden_size, parallel, device));
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
