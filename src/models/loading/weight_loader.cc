#include "inferx/models/loading/weight_loader.h"

#include <cstring>
#include <iterator>
#include <optional>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include "inferx/config/parallel_config.h"
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

/// \brief Splits interleaved row pairs into block-contiguous [A | B] halves.
///
/// Qwen3-Next interleaves [q_h | gate_h] per head and gpt-oss interleaves
/// gate/up rows per row; both become the block layout the packed kernels
/// expect. `block` is the interleave granularity in rows.
StatusOr<Tensor> DeinterleaveRows(const Tensor& host, int64_t block) {
  if (host.Rank() != 2 || host.Dim(0) % (2 * block) != 0) {
    return InvalidArgumentError("cannot de-interleave ", host.GetShape().ToString(),
                                " at block ", block);
  }
  const int64_t rows = host.Dim(0), cols = host.Dim(1);
  const int64_t pairs = rows / (2 * block);
  const int64_t elem = DataTypeByteSize(host.GetDataType(), 1);
  INFERX_ASSIGN_OR_RETURN(auto out,
                          Tensor::Empty(host.GetDataType(), Shape({rows, cols}), DeviceId::Cpu()));
  for (int64_t p = 0; p < pairs; ++p) {
    for (int64_t b = 0; b < block; ++b) {
      const auto* from_a = host.Bytes() + ((2 * p) * block + b) * cols * elem;
      const auto* from_b = host.Bytes() + ((2 * p + 1) * block + b) * cols * elem;
      auto* to_a = out.Bytes() + (p * block + b) * cols * elem;
      auto* to_b = out.Bytes() + (pairs * block + p * block + b) * cols * elem;
      std::memmove(to_a, from_a, static_cast<size_t>(cols * elem));
      std::memmove(to_b, from_b, static_cast<size_t>(cols * elem));
    }
  }
  return out;
}

/// \brief Zero-pads a rank-2 tensor's columns on the right.
StatusOr<Tensor> PadColumnsZero(const Tensor& host, int64_t padded_cols) {
  if (host.Rank() != 2 || padded_cols < host.Dim(1)) {
    return InvalidArgumentError("cannot pad ", host.GetShape().ToString(), " to ", padded_cols);
  }
  if (padded_cols == host.Dim(1)) return host;
  const int64_t rows = host.Dim(0), cols = host.Dim(1);
  const int64_t elem = DataTypeByteSize(host.GetDataType(), 1);
  INFERX_ASSIGN_OR_RETURN(
      auto out, Tensor::Empty(host.GetDataType(), Shape({rows, padded_cols}), DeviceId::Cpu()));
  std::memset(out.Bytes(), 0, out.NBytes());
  for (int64_t r = 0; r < rows; ++r) {
    std::memmove(out.Bytes() + r * padded_cols * elem, host.Bytes() + r * cols * elem,
                 static_cast<size_t>(cols * elem));
  }
  return out;
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
    const models::Checkpoint& checkpoint, const std::string& prefix,
    const models::WeightNames& names, const models::WeightLayout& layout,
    const components::AttentionConfig& a, const ParallelConfig& parallel, int64_t hidden,
    DeviceId device) {
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
    std::string_view checkpoint_name;
    int64_t full_rows;
    int64_t begin;
    int64_t rows;
  };
  const Part parts[] = {
      {names.q, total_query_rows, query_begin, geometry.query_rows},
      {names.k, total_kv_rows, kv_begin, geometry.kv_rows},
      {names.v, total_kv_rows, kv_begin, geometry.kv_rows},
  };

  std::optional<Tensor> query_view, key_view, value_view;
  std::optional<Tensor>* views[3] = {&query_view, &key_view, &value_view};
  std::optional<Tensor> fused;
  if (layout.qkv == models::QkvLayout::kFused) {
    if (names.qkv.empty()) return InvalidArgumentError("fused QKV requires a checkpoint name");
    INFERX_ASSIGN_OR_RETURN(
        fused, checkpoint.FindHostBf16(prefix + std::string(names.qkv) + ".weight",
                                       Shape({total_query_rows + 2 * total_kv_rows, hidden})));
  }
  int64_t source_offset = 0;
  int64_t offset = 0;
  for (size_t i = 0; i < std::size(parts); ++i) {
    const auto& part = parts[i];
    auto source = [&]() -> StatusOr<Tensor> {
      if (fused) return fused->Slice(source_offset, source_offset + part.full_rows);
      return checkpoint.FindHostBf16(prefix + std::string(part.checkpoint_name) + ".weight",
                                     Shape({part.full_rows, hidden}));
    };
    INFERX_ASSIGN_OR_RETURN(auto host, source());
    if (i == 0 && a.output_gate == components::OutputGate::kSigmoid) {
      // Qwen3-Next stores [q_h | gate_h] per head; reorder once here so the
      // packed projection splits [query | gate | key | value] in blocks.
      INFERX_ASSIGN_OR_RETURN(auto split, DeinterleaveRows(host, a.head_dim));
      host = std::move(split);
    }
    INFERX_ASSIGN_OR_RETURN(auto shard, host.Slice(part.begin, part.begin + part.rows));
    source_offset += part.full_rows;
    INFERX_ASSIGN_OR_RETURN(auto dst, packed.Slice(offset, offset + part.rows));
    INFERX_RETURN_IF_ERROR(shard.CopyTo(dst));
    *views[i] = std::move(dst);
    offset += part.rows;
  }

  // Packed bias [query (| gate) | key | value], mirroring the weight rows.
  std::optional<Tensor> qkv_bias;
  if (a.qkv_bias) {
    INFERX_ASSIGN_OR_RETURN(
        auto bias, Tensor::Empty(DataType::kBFloat16, Shape({packed_rows}), DeviceId::Cpu()));
    int64_t bias_offset = 0;
    const std::string_view bias_names[3] = {names.q, names.k, names.v};
    for (size_t i = 0; i < std::size(parts); ++i) {
      const auto& part = parts[i];
      INFERX_ASSIGN_OR_RETURN(
          auto host,
          checkpoint.FindHostBf16(prefix + std::string(bias_names[i]) + ".bias",
                                  Shape({part.full_rows})));
      if (i == 0 && a.output_gate == components::OutputGate::kSigmoid) {
        // The doubled query bias interleaves [q_h | gate_h] per element.
        INFERX_ASSIGN_OR_RETURN(auto two_d, host.Reshape(Shape({part.full_rows, 1})));
        INFERX_ASSIGN_OR_RETURN(auto split, DeinterleaveRows(two_d, 1));
        INFERX_ASSIGN_OR_RETURN(auto flat, split.Reshape(Shape({part.full_rows})));
        host = std::move(flat);
      }
      INFERX_ASSIGN_OR_RETURN(auto shard, host.Slice(part.begin, part.begin + part.rows));
      INFERX_ASSIGN_OR_RETURN(auto dst, bias.Slice(bias_offset, bias_offset + part.rows));
      INFERX_RETURN_IF_ERROR(shard.CopyTo(dst));
      bias_offset += part.rows;
    }
    INFERX_ASSIGN_OR_RETURN(auto bias_dev, bias.To(device));
    qkv_bias = std::move(bias_dev);
  }
  const int64_t total_query_width = a.query_heads * a.head_dim;  // Undoubled: o_proj input.
  INFERX_ASSIGN_OR_RETURN(auto output,
                          LoadRowParallel(checkpoint, prefix + std::string(names.o) + ".weight",
                                          hidden, total_query_width, parallel, device));
  std::optional<Tensor> query_norm, key_norm;
  if (a.qk_norm) {
    INFERX_ASSIGN_OR_RETURN(query_norm,
                            Weight(checkpoint, prefix + std::string(names.q_norm) + ".weight",
                                   {a.head_dim}, device));
    INFERX_ASSIGN_OR_RETURN(key_norm,
                            Weight(checkpoint, prefix + std::string(names.k_norm) + ".weight",
                                   {a.head_dim}, device));
  }
  std::optional<Tensor> output_bias;
  if (a.output_bias && checkpoint.Contains(prefix + std::string(names.o) + ".bias")) {
    INFERX_ASSIGN_OR_RETURN(auto o_bias,
                            Weight(checkpoint, prefix + std::string(names.o) + ".bias",
                                   {hidden}, device));
    INFERX_ASSIGN_OR_RETURN(auto o_bias_2d, o_bias.Reshape(Shape({1, hidden})));
    output_bias = std::move(o_bias_2d);
  }
  std::optional<Tensor> sinks;
  if (a.sinks) {
    INFERX_ASSIGN_OR_RETURN(auto sink_weights,
                            Weight(checkpoint, prefix + std::string(names.sinks),
                                   {a.query_heads}, device));
    sinks = std::move(sink_weights);
  }
  return components::AttentionWeights{
      std::move(packed),
      components::LinearWeights{std::move(*query_view), std::nullopt},
      components::LinearWeights{std::move(*key_view), std::nullopt},
      components::LinearWeights{std::move(*value_view), std::nullopt},
      components::LinearWeights{std::move(output), std::nullopt},
      std::move(query_norm),
      std::move(key_norm),
      std::move(qkv_bias),
      std::move(output_bias),
      std::move(sinks)};
}

/// \brief Loads a SwiGLU block: fused gate|up (MergedColumnParallelLinear --
///        one [2 * shard, hidden] allocation, per-rank row slices uploaded
///        directly) plus a RowParallel down projection.
StatusOr<components::SwiGluWeights> SwiGlu(const models::Checkpoint& checkpoint,
                                           const std::string& gate_name,
                                           const std::string& up_name,
                                           const std::string& down_name, int64_t hidden,
                                           int64_t total_intermediate,
                                           const ParallelConfig& parallel, DeviceId device) {
  INFERX_ASSIGN_OR_RETURN(auto shard, components::ShardDim(total_intermediate, parallel));
  const int64_t rows = shard.size;
  INFERX_ASSIGN_OR_RETURN(
      auto packed, Tensor::Empty(DataType::kBFloat16, Shape({2 * rows, hidden}), device));
  const struct {
    std::string name;
    int64_t offset;
  } parts[] = {{gate_name, 0}, {up_name, rows}};
  for (const auto& part : parts) {
    INFERX_ASSIGN_OR_RETURN(
        auto host,
        checkpoint.FindHostBf16(part.name + ".weight", Shape({total_intermediate, hidden})));
    INFERX_ASSIGN_OR_RETURN(auto slice, host.Slice(shard.begin, shard.begin + rows));
    INFERX_ASSIGN_OR_RETURN(auto dst, packed.Slice(part.offset, part.offset + rows));
    INFERX_RETURN_IF_ERROR(slice.CopyTo(dst));
  }
  INFERX_ASSIGN_OR_RETURN(Tensor gate_view, packed.Slice(0, rows));
  INFERX_ASSIGN_OR_RETURN(Tensor up_view, packed.Slice(rows, 2 * rows));
  INFERX_ASSIGN_OR_RETURN(auto down, LoadRowParallel(checkpoint, down_name + ".weight", hidden,
                                                     total_intermediate, parallel, device));
  return components::SwiGluWeights{
      std::move(packed), components::LinearWeights{std::move(gate_view), std::nullopt},
      components::LinearWeights{std::move(up_view), std::nullopt},
      components::LinearWeights{std::move(down), std::nullopt}};
}

/// \brief Loads MLA projections with the layout the kernels assume.
///
/// q_b rows reorder per head [nope | rope] -> [rope | nope]; kv_a splits into
/// its rope slice and latent; o_proj gains zero columns for the padded V tail.
StatusOr<components::MlaWeights> LoadMlaWeights(const models::Checkpoint& checkpoint,
                                                const std::string& prefix,
                                                const models::WeightNames& names,
                                                const components::MlaConfig& a, int64_t hidden,
                                                DeviceId device) {
  const int64_t head_dim = a.head_dim();
  INFERX_ASSIGN_OR_RETURN(auto q_a, Weight(checkpoint, prefix + std::string(names.mla_q_a) + ".weight",
                                           {a.q_lora_rank, hidden}, device));
  INFERX_ASSIGN_OR_RETURN(auto q_a_norm,
                          Weight(checkpoint, prefix + std::string(names.mla_q_a_norm) + ".weight",
                                 {a.q_lora_rank}, device));
  INFERX_ASSIGN_OR_RETURN(
      auto q_b_host,
      checkpoint.FindHostBf16(prefix + std::string(names.mla_q_b) + ".weight",
                             Shape({a.query_heads * head_dim, a.q_lora_rank})));
  // Per head, move the rope slice ahead of the nope slice.
  INFERX_ASSIGN_OR_RETURN(auto q_b_reordered,
                          Tensor::Empty(DataType::kBFloat16, q_b_host.GetShape(), DeviceId::Cpu()));
  for (int64_t h = 0; h < a.query_heads; ++h) {
    INFERX_ASSIGN_OR_RETURN(auto head, q_b_host.Slice(h * head_dim, (h + 1) * head_dim));
    INFERX_ASSIGN_OR_RETURN(auto nope, head.Slice(0, a.qk_nope_head_dim));
    INFERX_ASSIGN_OR_RETURN(auto rope, head.Slice(a.qk_nope_head_dim, head_dim));
    INFERX_ASSIGN_OR_RETURN(auto dst, q_b_reordered.Slice(h * head_dim, (h + 1) * head_dim));
    INFERX_ASSIGN_OR_RETURN(auto dst_rope, dst.Slice(0, a.qk_rope_head_dim));
    INFERX_ASSIGN_OR_RETURN(auto dst_nope, dst.Slice(a.qk_rope_head_dim, head_dim));
    INFERX_RETURN_IF_ERROR(rope.CopyTo(dst_rope));
    INFERX_RETURN_IF_ERROR(nope.CopyTo(dst_nope));
  }
  INFERX_ASSIGN_OR_RETURN(auto q_b, q_b_reordered.To(device));
  INFERX_ASSIGN_OR_RETURN(
      auto kv_a,
      checkpoint.FindHostBf16(prefix + std::string(names.mla_kv_a) + ".weight",
                             Shape({a.kv_lora_rank + a.qk_rope_head_dim, hidden})));
  INFERX_ASSIGN_OR_RETURN(auto latent_host, kv_a.Slice(0, a.kv_lora_rank));
  INFERX_ASSIGN_OR_RETURN(auto rope_host, kv_a.Slice(a.kv_lora_rank, a.kv_lora_rank + a.qk_rope_head_dim));
  INFERX_ASSIGN_OR_RETURN(auto kv_latent, latent_host.To(device));
  INFERX_ASSIGN_OR_RETURN(auto k_rope, rope_host.To(device));
  INFERX_ASSIGN_OR_RETURN(auto kv_a_norm,
                          Weight(checkpoint, prefix + std::string(names.mla_kv_a_norm) + ".weight",
                                 {a.kv_lora_rank}, device));
  INFERX_ASSIGN_OR_RETURN(
      auto kv_b, Weight(checkpoint, prefix + std::string(names.mla_kv_b) + ".weight",
                        {a.query_heads * (a.qk_nope_head_dim + a.v_head_dim), a.kv_lora_rank},
                        device));
  INFERX_ASSIGN_OR_RETURN(
      auto o_host, checkpoint.FindHostBf16(prefix + std::string(names.o) + ".weight",
                                           Shape({hidden, a.query_heads * a.v_head_dim})));
  INFERX_ASSIGN_OR_RETURN(auto o_padded, PadColumnsZero(o_host, a.query_heads * head_dim));
  INFERX_ASSIGN_OR_RETURN(auto output, o_padded.To(device));
  return components::MlaWeights{std::move(q_a),       std::move(q_a_norm),  std::move(q_b),
                                std::move(k_rope),    std::move(kv_latent), std::move(kv_a_norm),
                                std::move(kv_b),      std::move(output)};
}

/// \brief Loads the Gated DeltaNet projections and float32 gates.
StatusOr<components::GdnWeights> LoadGdnWeights(const models::Checkpoint& checkpoint,
                                                const std::string& prefix,
                                                const models::WeightNames& names,
                                                const components::GatedDeltaNetConfig& a,
                                                int64_t hidden, DeviceId device) {
  const int64_t q_total = a.key_heads * a.key_dim, v_total = a.value_heads * a.value_dim;
  INFERX_ASSIGN_OR_RETURN(
      auto qkvz, Weight(checkpoint, prefix + std::string(names.gdn_qkvz) + ".weight",
                        {2 * q_total + 2 * v_total, hidden}, device));
  INFERX_ASSIGN_OR_RETURN(auto ba, Weight(checkpoint, prefix + std::string(names.gdn_ba) + ".weight",
                                          {2 * a.value_heads, hidden}, device));
  const int64_t conv_dim = 2 * q_total + v_total;
  INFERX_ASSIGN_OR_RETURN(
      auto conv_host,
      checkpoint.FindHostBf16(prefix + std::string(names.gdn_conv) + ".weight",
                             Shape({conv_dim, 1, a.conv_kernel_size})));
  INFERX_ASSIGN_OR_RETURN(auto conv2d, conv_host.Reshape(Shape({conv_dim, a.conv_kernel_size})));
  INFERX_ASSIGN_OR_RETURN(auto conv, conv2d.To(device));
  INFERX_ASSIGN_OR_RETURN(auto a_log, checkpoint.UploadF32(prefix + std::string(names.gdn_a_log) + ".weight",
                                                           {a.value_heads}, device));
  INFERX_ASSIGN_OR_RETURN(auto dt_bias,
                          checkpoint.UploadF32(prefix + std::string(names.gdn_dt_bias) + ".weight",
                                               {a.value_heads}, device));
  INFERX_ASSIGN_OR_RETURN(auto norm,
                          Weight(checkpoint, prefix + std::string(names.gdn_norm) + ".weight",
                                 {a.value_dim}, device));
  INFERX_ASSIGN_OR_RETURN(auto out, Weight(checkpoint, prefix + std::string(names.gdn_out) + ".weight",
                                           {hidden, v_total}, device));
  return components::GdnWeights{std::move(qkvz),  std::move(ba), std::move(conv),
                                std::move(a_log), std::move(dt_bias),
                                std::move(norm),   std::move(out)};
}

/// \brief Loads gpt-oss's fused, MXFP4-quantized experts into SwiGLU weights.
///
/// Rows interleave gate/up per row; dequantization and de-interleaving both
/// happen once on the host, leaving ordinary bf16 block-contiguous experts.
StatusOr<std::vector<components::SwiGluWeights>> LoadFusedMxExperts(
    const models::Checkpoint& checkpoint, const std::string& prefix,
    const models::WeightNames& names, const components::MoeConfig& m, int64_t hidden,
    DeviceId device) {
  const int64_t inter = m.intermediate_size;
  const auto blocks = checkpoint.Find(prefix + std::string(names.experts_gate_up_blocks) + ".weight");
  const auto scales = checkpoint.Find(prefix + std::string(names.experts_gate_up_scales) + ".weight");
  if (!blocks.has_value() || !scales.has_value()) {
    return NotFoundError("fused expert blocks or scales missing in checkpoint");
  }
  INFERX_ASSIGN_OR_RETURN(
      auto bias,
      checkpoint.FindHostBf16(prefix + std::string(names.experts_gate_up_bias) + ".weight",
                             Shape({m.num_experts, 2 * inter})));
  const auto down_blocks =
      checkpoint.Find(prefix + std::string(names.experts_down_blocks) + ".weight");
  const auto down_scales =
      checkpoint.Find(prefix + std::string(names.experts_down_scales) + ".weight");
  if (!down_blocks.has_value() || !down_scales.has_value()) {
    return NotFoundError("fused expert down blocks or scales missing in checkpoint");
  }
  INFERX_ASSIGN_OR_RETURN(
      auto down_bias,
      checkpoint.FindHostBf16(prefix + std::string(names.experts_down_bias) + ".weight",
                             Shape({m.num_experts, hidden})));
  std::vector<components::SwiGluWeights> experts;
  experts.reserve(m.num_experts);
  for (int64_t e = 0; e < m.num_experts; ++e) {
    INFERX_ASSIGN_OR_RETURN(auto gate_up_host,
                            checkpoint.DequantMxToBf16(
                                blocks->Slice(e * 2 * inter, (e + 1) * 2 * inter).value(),
                                scales->Slice(e * 2 * inter, (e + 1) * 2 * inter).value(),
                                Shape({2 * inter, hidden})));
    INFERX_ASSIGN_OR_RETURN(auto gate_up, DeinterleaveRows(gate_up_host, 1));
    INFERX_ASSIGN_OR_RETURN(auto gate_up_dev, gate_up.To(device));
    INFERX_ASSIGN_OR_RETURN(auto gate_rows, gate_up_dev.Slice(0, inter));
    INFERX_ASSIGN_OR_RETURN(auto up_rows, gate_up_dev.Slice(inter, 2 * inter));
    INFERX_ASSIGN_OR_RETURN(auto bias_rows, bias.Slice(e, e + 1));
    INFERX_ASSIGN_OR_RETURN(auto bias_split, DeinterleaveRows(bias_rows, 1));
    INFERX_ASSIGN_OR_RETURN(auto packed_bias, bias_split.Reshape(Shape({2 * inter})));
    INFERX_ASSIGN_OR_RETURN(auto packed_bias_dev, packed_bias.To(device));
    INFERX_ASSIGN_OR_RETURN(
        auto down_host,
        checkpoint.DequantMxToBf16(down_blocks->Slice(e * hidden, (e + 1) * hidden).value(),
                                   down_scales->Slice(e * hidden, (e + 1) * hidden).value(),
                                   Shape({hidden, inter})));
    INFERX_ASSIGN_OR_RETURN(auto down, down_host.To(device));
    INFERX_ASSIGN_OR_RETURN(auto down_bias_rows, down_bias.Slice(e, e + 1));
    INFERX_ASSIGN_OR_RETURN(auto down_bias_dev,
                            down_bias_rows.Reshape(Shape({hidden})).value().To(device));
    experts.push_back(components::SwiGluWeights{
        std::move(gate_up_dev),
        components::LinearWeights{std::move(gate_rows), std::nullopt},
        components::LinearWeights{std::move(up_rows), std::nullopt},
        components::LinearWeights{std::move(down), std::nullopt},
        std::move(packed_bias_dev), std::move(down_bias_dev)});
  }
  return experts;
}

StatusOr<components::DecoderLayerWeights> LoadDecoderLayer(
    const models::Checkpoint& checkpoint, const components::DecoderLayerConfig& config,
    const models::WeightNames& names, const models::WeightLayout& layout,
    const std::string& prefix, int64_t hidden, const ParallelConfig& parallel,
    DeviceId device) {
  INFERX_ASSIGN_OR_RETURN(
      auto input_norm,
      Weight(checkpoint, prefix + std::string(names.attn_norm) + ".weight", {hidden}, device));
  INFERX_ASSIGN_OR_RETURN(
      auto post_mixer_norm,
      Weight(checkpoint, prefix + std::string(names.ffn_norm) + ".weight", {hidden}, device));
  std::optional<Tensor> mixer_out_norm, feed_forward_out_norm;
  if (config.residual == components::ResidualStyle::kOutputNorm) {
    if (names.attn_out_norm.empty() || names.ffn_out_norm.empty()) {
      return InvalidArgumentError("output-normalized layers need their norm names");
    }
    INFERX_ASSIGN_OR_RETURN(
        auto mixer_norm,
        Weight(checkpoint, prefix + std::string(names.attn_out_norm) + ".weight", {hidden}, device));
    INFERX_ASSIGN_OR_RETURN(
        auto ffn_norm_out,
        Weight(checkpoint, prefix + std::string(names.ffn_out_norm) + ".weight", {hidden}, device));
    mixer_out_norm = std::move(mixer_norm);
    feed_forward_out_norm = std::move(ffn_norm_out);
  }
  std::optional<std::variant<components::AttentionWeights, components::MlaWeights,
                             components::GdnWeights>>
      attn;
  if (const auto* a = std::get_if<components::AttentionConfig>(&config.mixer)) {
    INFERX_ASSIGN_OR_RETURN(auto weights,
                            LoadAttentionWeights(checkpoint, prefix, names, layout, *a, parallel,
                                                 hidden, device));
    attn = std::move(weights);
  } else if (const auto* m = std::get_if<components::MlaConfig>(&config.mixer)) {
    INFERX_ASSIGN_OR_RETURN(auto weights, LoadMlaWeights(checkpoint, prefix, names, *m, hidden,
                                                          device));
    attn = std::move(weights);
  } else {
    const auto& g = std::get<components::GatedDeltaNetConfig>(config.mixer);
    INFERX_ASSIGN_OR_RETURN(auto weights, LoadGdnWeights(checkpoint, prefix, names, g, hidden,
                                                          device));
    attn = std::move(weights);
  }
  if (const auto* dense = std::get_if<components::SwiGluConfig>(&config.feed_forward)) {
    INFERX_ASSIGN_OR_RETURN(
        auto ffn, SwiGlu(checkpoint, prefix + std::string(names.gate),
                         prefix + std::string(names.up), prefix + std::string(names.down),
                         hidden, dense->intermediate_size, parallel, device));
    return components::DecoderLayerWeights{std::move(input_norm), std::move(post_mixer_norm),
                                           std::move(mixer_out_norm),
                                           std::move(feed_forward_out_norm),
                                           std::move(*attn), std::move(ffn)};
  }
  const auto& m = std::get<components::MoeConfig>(config.feed_forward);
  INFERX_ASSIGN_OR_RETURN(auto router,
                          Weight(checkpoint, prefix + std::string(names.router) + ".weight",
                                 {m.num_experts, hidden}, device));
  std::optional<Tensor> router_bias;
  if (m.has_router_bias && checkpoint.Contains(prefix + std::string(names.router) + ".bias")) {
    INFERX_ASSIGN_OR_RETURN(auto bias,
                            Weight(checkpoint, prefix + std::string(names.router) + ".bias",
                                   {m.num_experts}, device));
    router_bias = std::move(bias);
  }
  std::optional<Tensor> correction_bias;
  if (m.has_correction_bias) {
    INFERX_ASSIGN_OR_RETURN(auto bias,
                            checkpoint.UploadF32(
                                prefix + std::string(names.correction_bias) + ".weight",
                                {m.num_experts}, device));
    correction_bias = std::move(bias);
  }
  std::vector<components::SwiGluWeights> experts;
  if (m.fused_mxfp4_experts) {
    INFERX_ASSIGN_OR_RETURN(experts,
                            LoadFusedMxExperts(checkpoint, prefix, names, m, hidden, device));
  }
  for (int64_t i = 0; i < m.num_experts && !m.fused_mxfp4_experts; ++i) {
    const auto expert_prefix =
        prefix + std::string(names.experts) + "." + std::to_string(i) + ".";
    INFERX_ASSIGN_OR_RETURN(
        auto expert,
        SwiGlu(checkpoint, expert_prefix + "gate_proj", expert_prefix + "up_proj",
               expert_prefix + "down_proj", hidden, m.intermediate_size, parallel, device));
    experts.push_back(std::move(expert));
  }
  std::optional<components::SwiGluWeights> shared_expert;
  std::optional<Tensor> shared_expert_gate;
  if (m.shared_intermediate_size > 0) {
    const auto shared_prefix = prefix + std::string(names.shared_expert) + ".";
    INFERX_ASSIGN_OR_RETURN(
        auto shared, SwiGlu(checkpoint, shared_prefix + "gate_proj",
                            shared_prefix + "up_proj", shared_prefix + "down_proj", hidden,
                            m.shared_intermediate_size, parallel, device));
    shared_expert = std::move(shared);
    if (m.gate_shared_expert) {
      INFERX_ASSIGN_OR_RETURN(
          auto gate, Weight(checkpoint, prefix + std::string(names.shared_expert_gate) + ".weight",
                            {1, hidden}, device));
      shared_expert_gate = std::move(gate);
    }
  }
  components::MoeWeights moe{std::move(router),       std::move(router_bias),
                             std::move(correction_bias), std::move(experts),
                             std::move(shared_expert),   std::move(shared_expert_gate)};
  return components::DecoderLayerWeights{std::move(input_norm), std::move(post_mixer_norm),
                                         std::move(mixer_out_norm),
                                         std::move(feed_forward_out_norm),
                                         std::move(*attn), std::move(moe)};
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
  INFERX_ASSIGN_OR_RETURN(auto host, checkpoint.FindHostBf16(name, Shape({vocab, hidden})));
  INFERX_ASSIGN_OR_RETURN(auto slice, host.Slice(shard.begin, shard.begin + shard.size));
  return slice.To(device);
}

StatusOr<DecoderWeights> LoadDecoderWeights(const models::Checkpoint& checkpoint,
                                            const DecoderConfig& config,
                                            const models::WeightNames& names,
                                            const models::WeightLayout& layout,
                                            const ParallelConfig& parallel, DeviceId device) {
  const auto& mc = config.model;
  INFERX_ASSIGN_OR_RETURN(auto token_embedding,
                          LoadVocabShard(checkpoint, std::string(names.embed) + ".weight",
                                         mc.vocab_size, mc.hidden_size, parallel, device));
  INFERX_ASSIGN_OR_RETURN(
      auto final_norm,
      Weight(checkpoint, std::string(names.final_norm) + ".weight", {mc.hidden_size}, device));
  std::vector<components::DecoderLayerWeights> blocks;
  for (size_t i = 0; i < config.blocks.size(); ++i) {
    INFERX_ASSIGN_OR_RETURN(
        auto block, LoadDecoderLayer(checkpoint, config.blocks[i], names, layout,
                                     std::string(names.layers) + "." + std::to_string(i) + ".",
                                     mc.hidden_size, parallel, device));
    blocks.push_back(std::move(block));
  }
  return DecoderWeights{std::move(token_embedding), std::move(final_norm), std::move(blocks)};
}

}  // namespace inferx::causal
