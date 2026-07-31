#include "inferx/model/qwen2.h"

#include <cmath>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

#include <cuda_runtime.h>

#include "absl/strings/str_cat.h"
#include "inferx/core/cuda_utils.h"
#include "inferx/core/device_buffer.h"
#include "inferx/core/shape.h"
#include "inferx/core/tensor_view.h"
#include "inferx/kernels/gemm.h"
#include "inferx/kernels/layers.h"

namespace inferx::model {
namespace {

/// Copies one checkpoint tensor to the device and returns a view over it.
///
/// The source is a borrowed host tensor over the checkpoint's mapping, so this
/// is the point where 6 GB actually crosses PCIe -- and the point where the
/// pages get faulted in, one tensor at a time, rather than all at once.
StatusOr<TensorView> Upload(const Tensor& host, std::vector<DeviceBuffer>* keep) {
  INFERX_ASSIGN_OR_RETURN(
      DeviceBuffer buf,
      DeviceBuffer::Allocate(static_cast<size_t>(host.nbytes()),
                             DeviceId::Cuda(0)));

  INFERX_CUDA_RETURN_IF_ERROR(cudaMemcpy(buf.data(), host.data(),
                                         static_cast<size_t>(host.nbytes()),
                                         cudaMemcpyHostToDevice));

  keep->push_back(std::move(buf));

  return TensorView::Create(keep->back().data(), host.dtype(), host.shape(),
                            DeviceId::Cuda(0));
}

/// Weights for one decoder block. Views into buffers owned by Impl.
struct LayerWeights {
  TensorView input_norm;
  TensorView q_w, k_w, v_w, o_w;
  TensorView q_b, k_b, v_b;  // undefined when the architecture has no bias
  TensorView post_norm;
  TensorView gate_w, up_w, down_w;
};

/// A scratch buffer plus the view over it, so activations are allocated once
/// and reshaped per call rather than per layer.
struct Scratch {
  DeviceBuffer buf;

  StatusOr<TensorView> View(DataType dtype, const Shape& shape) const {
    return TensorView::Create(buf.data(), dtype, shape, DeviceId::Cuda(0));
  }
};

}  // namespace

struct Qwen2Model::Impl {
  ModelConfig config;

  std::vector<DeviceBuffer> weight_buffers;
  std::vector<LayerWeights> layers;
  TensorView embed;       // [vocab, hidden] -- also the LM head when tied
  TensorView final_norm;  // [hidden]
  TensorView lm_head;     // == embed when tie_word_embeddings

  kernels::CublasLtGemm gemm;

  // Activation scratch, sized for the longest prompt seen so far and grown on
  // demand. Reused across layers: the stack is a chain of same-shaped buffers,
  // so a fresh allocation per layer would be 36x the traffic for nothing.
  int64_t capacity_tokens = 0;
  Scratch hidden, residual, normed;
  Scratch q, k, v, attn_out;
  Scratch gate, up;
  Scratch logits;
  DeviceBuffer positions;
  DeviceBuffer token_ids;

  size_t weight_bytes = 0;

  explicit Impl(kernels::CublasLtGemm g) : gemm(std::move(g)) {}

  // M3: the paged cache and the per-step index buffers that address it.
  std::unique_ptr<KvBlockPool> pool;
  DeviceBuffer slots_buf, seq_of_token_buf, block_table_buf;
  int64_t block_table_capacity = 0;

  Status EnsureCapacity(int64_t tokens);
  Status RunForward(const std::vector<int32_t>& ids, int64_t* out_tokens);
  Status RunPagedForward(const ForwardBatch& batch);
};

Status Qwen2Model::Impl::EnsureCapacity(int64_t tokens) {
  if (tokens <= capacity_tokens) return OkStatus();

  const int64_t h = config.hidden_size;
  const int64_t qd = config.q_dim();
  const int64_t kvd = config.kv_dim();
  const int64_t inter = config.intermediate_size;
  const int64_t vocab = config.vocab_size;

  const auto alloc = [&](Scratch* s, int64_t elems, size_t elem_size) -> Status {
    INFERX_ASSIGN_OR_RETURN(
        s->buf, DeviceBuffer::Allocate(static_cast<size_t>(elems) * elem_size,
                                       DeviceId::Cuda(0)));
    return OkStatus();
  };

  constexpr size_t kBf16 = 2;

  INFERX_RETURN_IF_ERROR(alloc(&hidden, tokens * h, kBf16));
  INFERX_RETURN_IF_ERROR(alloc(&residual, tokens * h, kBf16));
  INFERX_RETURN_IF_ERROR(alloc(&normed, tokens * h, kBf16));
  INFERX_RETURN_IF_ERROR(alloc(&q, tokens * qd, kBf16));
  INFERX_RETURN_IF_ERROR(alloc(&k, tokens * kvd, kBf16));
  INFERX_RETURN_IF_ERROR(alloc(&v, tokens * kvd, kBf16));
  INFERX_RETURN_IF_ERROR(alloc(&attn_out, tokens * qd, kBf16));
  INFERX_RETURN_IF_ERROR(alloc(&gate, tokens * inter, kBf16));
  INFERX_RETURN_IF_ERROR(alloc(&up, tokens * inter, kBf16));
  INFERX_RETURN_IF_ERROR(alloc(&logits, tokens * vocab, kBf16));

  INFERX_ASSIGN_OR_RETURN(
      positions, DeviceBuffer::Allocate(
                     static_cast<size_t>(tokens) * sizeof(int32_t),
                     DeviceId::Cuda(0)));
  INFERX_ASSIGN_OR_RETURN(
      token_ids, DeviceBuffer::Allocate(
                     static_cast<size_t>(tokens) * sizeof(int32_t),
                     DeviceId::Cuda(0)));

  capacity_tokens = tokens;
  return OkStatus();
}

Status Qwen2Model::Impl::RunForward(const std::vector<int32_t>& ids,
                                    int64_t* out_tokens) {
  const int64_t tokens = static_cast<int64_t>(ids.size());
  const int64_t h = config.hidden_size;
  const int64_t heads = config.num_attention_heads;
  const int64_t kv_heads = config.num_key_value_heads;
  const int64_t hd = config.head_dim;
  const int64_t inter = config.intermediate_size;

  INFERX_RETURN_IF_ERROR(EnsureCapacity(tokens));

  *out_tokens = tokens;

  // Positions are 0..n-1: there is no cache to continue from at M2.
  std::vector<int32_t> pos(static_cast<size_t>(tokens));
  for (int64_t i = 0; i < tokens; ++i) pos[static_cast<size_t>(i)] =
      static_cast<int32_t>(i);

  INFERX_CUDA_RETURN_IF_ERROR(
      cudaMemcpy(positions.data(), pos.data(),
                 pos.size() * sizeof(int32_t), cudaMemcpyHostToDevice));
  INFERX_CUDA_RETURN_IF_ERROR(
      cudaMemcpy(token_ids.data(), ids.data(), ids.size() * sizeof(int32_t),
                 cudaMemcpyHostToDevice));

  INFERX_ASSIGN_OR_RETURN(
      const TensorView ids_v,
      TensorView::Create(token_ids.data(), DataType::kInt32, Shape({tokens}),
                         DeviceId::Cuda(0)));
  INFERX_ASSIGN_OR_RETURN(
      const TensorView pos_v,
      TensorView::Create(positions.data(), DataType::kInt32, Shape({tokens}),
                         DeviceId::Cuda(0)));

  const Shape hidden_shape({tokens, h});
  INFERX_ASSIGN_OR_RETURN(const TensorView x,
                          hidden.View(DataType::kBFloat16, hidden_shape));
  INFERX_ASSIGN_OR_RETURN(const TensorView resid,
                          residual.View(DataType::kBFloat16, hidden_shape));
  INFERX_ASSIGN_OR_RETURN(const TensorView norm,
                          normed.View(DataType::kBFloat16, hidden_shape));

  INFERX_ASSIGN_OR_RETURN(
      const TensorView qv,
      q.View(DataType::kBFloat16, Shape({tokens, config.q_dim()})));
  INFERX_ASSIGN_OR_RETURN(
      const TensorView kv_,
      k.View(DataType::kBFloat16, Shape({tokens, config.kv_dim()})));
  INFERX_ASSIGN_OR_RETURN(
      const TensorView vv,
      v.View(DataType::kBFloat16, Shape({tokens, config.kv_dim()})));
  INFERX_ASSIGN_OR_RETURN(
      const TensorView av,
      attn_out.View(DataType::kBFloat16, Shape({tokens, config.q_dim()})));

  // The same buffers seen as [tokens, heads, head_dim] for the attention
  // kernels. Same bytes, different shape -- the projections produce a flat
  // [tokens, heads*head_dim] and attention wants the heads split out.
  INFERX_ASSIGN_OR_RETURN(
      const TensorView q3,
      q.View(DataType::kBFloat16, Shape({tokens, heads, hd})));
  INFERX_ASSIGN_OR_RETURN(
      const TensorView k3,
      k.View(DataType::kBFloat16, Shape({tokens, kv_heads, hd})));
  INFERX_ASSIGN_OR_RETURN(
      const TensorView v3,
      v.View(DataType::kBFloat16, Shape({tokens, kv_heads, hd})));
  INFERX_ASSIGN_OR_RETURN(
      const TensorView a3,
      attn_out.View(DataType::kBFloat16, Shape({tokens, heads, hd})));

  INFERX_ASSIGN_OR_RETURN(
      const TensorView gate_v,
      gate.View(DataType::kBFloat16, Shape({tokens, inter})));
  INFERX_ASSIGN_OR_RETURN(
      const TensorView up_v,
      up.View(DataType::kBFloat16, Shape({tokens, inter})));

  const float attn_scale = 1.0f / std::sqrt(static_cast<float>(hd));

  INFERX_RETURN_IF_ERROR(kernels::EmbeddingLookup(embed, ids_v, x));

  for (const LayerWeights& layer : layers) {
    // --- attention block -------------------------------------------------
    INFERX_CUDA_RETURN_IF_ERROR(cudaMemcpy(resid.Data(), x.Data(),
                                           static_cast<size_t>(x.NBytes()),
                                           cudaMemcpyDeviceToDevice));

    INFERX_RETURN_IF_ERROR(kernels::RmsNorm(x, layer.input_norm, norm,
                                            static_cast<float>(
                                                config.rms_norm_eps)));

    INFERX_RETURN_IF_ERROR(gemm.LinearBF16(norm, layer.q_w, qv));
    INFERX_RETURN_IF_ERROR(gemm.LinearBF16(norm, layer.k_w, kv_));
    INFERX_RETURN_IF_ERROR(gemm.LinearBF16(norm, layer.v_w, vv));

    if (config.attention_bias) {
      INFERX_RETURN_IF_ERROR(kernels::AddBiasInPlace(qv, layer.q_b));
      INFERX_RETURN_IF_ERROR(kernels::AddBiasInPlace(kv_, layer.k_b));
      INFERX_RETURN_IF_ERROR(kernels::AddBiasInPlace(vv, layer.v_b));
    }

    // RoPE before attention and after the bias: the bias is part of the
    // projection, and rotating a biased Q is what the reference does.
    INFERX_RETURN_IF_ERROR(kernels::RotaryEmbedding(
        q3, k3, pos_v, static_cast<float>(config.rope_theta)));

    INFERX_RETURN_IF_ERROR(kernels::Attention(q3, k3, v3, a3, attn_scale));

    INFERX_RETURN_IF_ERROR(gemm.LinearBF16(av, layer.o_w, x));
    INFERX_RETURN_IF_ERROR(kernels::AddInPlace(x, resid));

    // --- feed-forward block ----------------------------------------------
    INFERX_CUDA_RETURN_IF_ERROR(cudaMemcpy(resid.Data(), x.Data(),
                                           static_cast<size_t>(x.NBytes()),
                                           cudaMemcpyDeviceToDevice));

    INFERX_RETURN_IF_ERROR(kernels::RmsNorm(x, layer.post_norm, norm,
                                            static_cast<float>(
                                                config.rms_norm_eps)));

    INFERX_RETURN_IF_ERROR(gemm.LinearBF16(norm, layer.gate_w, gate_v));
    INFERX_RETURN_IF_ERROR(gemm.LinearBF16(norm, layer.up_w, up_v));
    INFERX_RETURN_IF_ERROR(kernels::SiluMul(gate_v, up_v, gate_v));
    INFERX_RETURN_IF_ERROR(gemm.LinearBF16(gate_v, layer.down_w, x));
    INFERX_RETURN_IF_ERROR(kernels::AddInPlace(x, resid));
  }

  INFERX_RETURN_IF_ERROR(kernels::RmsNorm(x, final_norm, norm,
                                          static_cast<float>(
                                              config.rms_norm_eps)));

  INFERX_ASSIGN_OR_RETURN(
      const TensorView logits_v,
      logits.View(DataType::kBFloat16, Shape({tokens, config.vocab_size})));

  INFERX_RETURN_IF_ERROR(gemm.LinearBF16(norm, lm_head, logits_v));

  INFERX_CUDA_RETURN_IF_ERROR(cudaDeviceSynchronize());

  return OkStatus();
}

Status ForwardBatch::Validate(int64_t vocab_size, int64_t total_slots) const {
  const int64_t n = num_tokens();

  if (n == 0) return InvalidArgumentError("batch has no tokens");

  const auto same_size = [&](const std::vector<int32_t>& v, const char* name) {
    return static_cast<int64_t>(v.size()) == n
               ? OkStatus()
               : InvalidArgumentError(name, " has ", v.size(), " entries but "
                                      "there are ", n, " tokens");
  };

  INFERX_RETURN_IF_ERROR(same_size(positions, "positions"));
  INFERX_RETURN_IF_ERROR(same_size(seq_of_token, "seq_of_token"));
  INFERX_RETURN_IF_ERROR(same_size(slots, "slots"));

  if (num_seqs <= 0 || max_blocks_per_seq <= 0) {
    return InvalidArgumentError("batch has num_seqs=", num_seqs,
                                " max_blocks_per_seq=", max_blocks_per_seq);
  }

  if (static_cast<int64_t>(block_table.size()) !=
      num_seqs * max_blocks_per_seq) {
    return InvalidArgumentError("block_table has ", block_table.size(),
                                " entries, expected ",
                                num_seqs * max_blocks_per_seq);
  }

  for (int64_t i = 0; i < n; ++i) {
    const size_t u = static_cast<size_t>(i);

    if (token_ids[u] < 0 || token_ids[u] >= vocab_size) {
      return InvalidArgumentError("token_ids[", i, "] = ", token_ids[u],
                                  " is outside the vocabulary");
    }
    if (positions[u] < 0) {
      return InvalidArgumentError("positions[", i, "] = ", positions[u],
                                  " is negative");
    }
    if (seq_of_token[u] < 0 || seq_of_token[u] >= num_seqs) {
      return InvalidArgumentError("seq_of_token[", i, "] = ", seq_of_token[u],
                                  " is outside [0, ", num_seqs, ")");
    }
    // A slot outside the pool would scatter this token's keys into whatever
    // follows the cache in device memory.
    if (slots[u] < 0 || slots[u] >= total_slots) {
      return InvalidArgumentError("slots[", i, "] = ", slots[u],
                                  " is outside [0, ", total_slots, ")");
    }
  }

  if (logits_indices.empty()) {
    return InvalidArgumentError("batch asks for no logits");
  }

  for (size_t i = 0; i < logits_indices.size(); ++i) {
    if (logits_indices[i] < 0 || logits_indices[i] >= n) {
      return InvalidArgumentError("logits_indices[", i, "] = ",
                                  logits_indices[i], " is outside [0, ", n,
                                  ")");
    }
  }

  return OkStatus();
}

Status Qwen2Model::Impl::RunPagedForward(const ForwardBatch& batch) {
  const int64_t tokens = batch.num_tokens();
  const int64_t h = config.hidden_size;
  const int64_t heads = config.num_attention_heads;
  const int64_t kv_heads = config.num_key_value_heads;
  const int64_t hd = config.head_dim;
  const int64_t inter = config.intermediate_size;

  INFERX_RETURN_IF_ERROR(EnsureCapacity(tokens));

  // Index buffers. Grown with the token capacity, except the block table, which
  // scales with sequences rather than tokens.
  const int64_t table_elems = batch.num_seqs * batch.max_blocks_per_seq;

  if (table_elems > block_table_capacity) {
    INFERX_ASSIGN_OR_RETURN(
        block_table_buf,
        DeviceBuffer::Allocate(static_cast<size_t>(table_elems) *
                                   sizeof(int32_t),
                               DeviceId::Cuda(0)));
    block_table_capacity = table_elems;
  }

  if (slots_buf.size() < static_cast<size_t>(tokens) * sizeof(int32_t)) {
    INFERX_ASSIGN_OR_RETURN(
        slots_buf, DeviceBuffer::Allocate(
                       static_cast<size_t>(tokens) * sizeof(int32_t),
                       DeviceId::Cuda(0)));
    INFERX_ASSIGN_OR_RETURN(
        seq_of_token_buf, DeviceBuffer::Allocate(
                              static_cast<size_t>(tokens) * sizeof(int32_t),
                              DeviceId::Cuda(0)));
  }

  const auto upload = [&](const DeviceBuffer& buf,
                          const std::vector<int32_t>& src) -> Status {
    INFERX_CUDA_RETURN_IF_ERROR(cudaMemcpy(buf.data(), src.data(),
                                           src.size() * sizeof(int32_t),
                                           cudaMemcpyHostToDevice));
    return OkStatus();
  };

  INFERX_RETURN_IF_ERROR(upload(token_ids, batch.token_ids));
  INFERX_RETURN_IF_ERROR(upload(positions, batch.positions));
  INFERX_RETURN_IF_ERROR(upload(slots_buf, batch.slots));
  INFERX_RETURN_IF_ERROR(upload(seq_of_token_buf, batch.seq_of_token));
  INFERX_RETURN_IF_ERROR(upload(block_table_buf, batch.block_table));

  const auto i32 = [&](const DeviceBuffer& buf,
                       const Shape& shape) -> StatusOr<TensorView> {
    return TensorView::Create(buf.data(), DataType::kInt32, shape,
                              DeviceId::Cuda(0));
  };

  INFERX_ASSIGN_OR_RETURN(const TensorView ids_v,
                          i32(token_ids, Shape({tokens})));
  INFERX_ASSIGN_OR_RETURN(const TensorView pos_v,
                          i32(positions, Shape({tokens})));
  INFERX_ASSIGN_OR_RETURN(const TensorView slots_v,
                          i32(slots_buf, Shape({tokens})));
  INFERX_ASSIGN_OR_RETURN(const TensorView seq_v,
                          i32(seq_of_token_buf, Shape({tokens})));
  INFERX_ASSIGN_OR_RETURN(
      const TensorView table_v,
      i32(block_table_buf,
          Shape({batch.num_seqs, batch.max_blocks_per_seq})));

  const Shape hidden_shape({tokens, h});
  INFERX_ASSIGN_OR_RETURN(const TensorView x,
                          hidden.View(DataType::kBFloat16, hidden_shape));
  INFERX_ASSIGN_OR_RETURN(const TensorView resid,
                          residual.View(DataType::kBFloat16, hidden_shape));
  INFERX_ASSIGN_OR_RETURN(const TensorView norm,
                          normed.View(DataType::kBFloat16, hidden_shape));

  INFERX_ASSIGN_OR_RETURN(
      const TensorView qv,
      q.View(DataType::kBFloat16, Shape({tokens, config.q_dim()})));
  INFERX_ASSIGN_OR_RETURN(
      const TensorView kv_,
      k.View(DataType::kBFloat16, Shape({tokens, config.kv_dim()})));
  INFERX_ASSIGN_OR_RETURN(
      const TensorView vv,
      v.View(DataType::kBFloat16, Shape({tokens, config.kv_dim()})));
  INFERX_ASSIGN_OR_RETURN(
      const TensorView av,
      attn_out.View(DataType::kBFloat16, Shape({tokens, config.q_dim()})));

  INFERX_ASSIGN_OR_RETURN(
      const TensorView q3,
      q.View(DataType::kBFloat16, Shape({tokens, heads, hd})));
  INFERX_ASSIGN_OR_RETURN(
      const TensorView k3,
      k.View(DataType::kBFloat16, Shape({tokens, kv_heads, hd})));
  INFERX_ASSIGN_OR_RETURN(
      const TensorView v3,
      v.View(DataType::kBFloat16, Shape({tokens, kv_heads, hd})));
  INFERX_ASSIGN_OR_RETURN(
      const TensorView a3,
      attn_out.View(DataType::kBFloat16, Shape({tokens, heads, hd})));

  INFERX_ASSIGN_OR_RETURN(
      const TensorView gate_v,
      gate.View(DataType::kBFloat16, Shape({tokens, inter})));
  INFERX_ASSIGN_OR_RETURN(
      const TensorView up_v,
      up.View(DataType::kBFloat16, Shape({tokens, inter})));

  const float attn_scale = 1.0f / std::sqrt(static_cast<float>(hd));

  INFERX_RETURN_IF_ERROR(kernels::EmbeddingLookup(embed, ids_v, x));

  for (int64_t layer_index = 0;
       layer_index < static_cast<int64_t>(layers.size()); ++layer_index) {
    const LayerWeights& layer = layers[static_cast<size_t>(layer_index)];

    INFERX_CUDA_RETURN_IF_ERROR(cudaMemcpy(resid.Data(), x.Data(),
                                           static_cast<size_t>(x.NBytes()),
                                           cudaMemcpyDeviceToDevice));

    INFERX_RETURN_IF_ERROR(kernels::RmsNorm(
        x, layer.input_norm, norm, static_cast<float>(config.rms_norm_eps)));

    INFERX_RETURN_IF_ERROR(gemm.LinearBF16(norm, layer.q_w, qv));
    INFERX_RETURN_IF_ERROR(gemm.LinearBF16(norm, layer.k_w, kv_));
    INFERX_RETURN_IF_ERROR(gemm.LinearBF16(norm, layer.v_w, vv));

    if (config.attention_bias) {
      INFERX_RETURN_IF_ERROR(kernels::AddBiasInPlace(qv, layer.q_b));
      INFERX_RETURN_IF_ERROR(kernels::AddBiasInPlace(kv_, layer.k_b));
      INFERX_RETURN_IF_ERROR(kernels::AddBiasInPlace(vv, layer.v_b));
    }

    INFERX_RETURN_IF_ERROR(kernels::RotaryEmbedding(
        q3, k3, pos_v, static_cast<float>(config.rope_theta)));

    // The cache is written *after* RoPE and *before* attention. Storing rotated
    // keys is what lets a cached key be reused at every later step without
    // re-rotating it -- the position it was rotated for is the position it
    // keeps.
    INFERX_ASSIGN_OR_RETURN(const TensorView k_cache,
                            pool->KeyCache(layer_index));
    INFERX_ASSIGN_OR_RETURN(const TensorView v_cache,
                            pool->ValueCache(layer_index));

    INFERX_RETURN_IF_ERROR(
        kernels::AppendToKvCache(k3, v3, k_cache, v_cache, slots_v));

    INFERX_RETURN_IF_ERROR(kernels::PagedAttention(
        q3, k_cache, v_cache, table_v, seq_v, pos_v, a3, attn_scale));

    INFERX_RETURN_IF_ERROR(gemm.LinearBF16(av, layer.o_w, x));
    INFERX_RETURN_IF_ERROR(kernels::AddInPlace(x, resid));

    INFERX_CUDA_RETURN_IF_ERROR(cudaMemcpy(resid.Data(), x.Data(),
                                           static_cast<size_t>(x.NBytes()),
                                           cudaMemcpyDeviceToDevice));

    INFERX_RETURN_IF_ERROR(kernels::RmsNorm(
        x, layer.post_norm, norm, static_cast<float>(config.rms_norm_eps)));

    INFERX_RETURN_IF_ERROR(gemm.LinearBF16(norm, layer.gate_w, gate_v));
    INFERX_RETURN_IF_ERROR(gemm.LinearBF16(norm, layer.up_w, up_v));
    INFERX_RETURN_IF_ERROR(kernels::SiluMul(gate_v, up_v, gate_v));
    INFERX_RETURN_IF_ERROR(gemm.LinearBF16(gate_v, layer.down_w, x));
    INFERX_RETURN_IF_ERROR(kernels::AddInPlace(x, resid));
  }

  INFERX_RETURN_IF_ERROR(kernels::RmsNorm(
      x, final_norm, norm, static_cast<float>(config.rms_norm_eps)));

  INFERX_ASSIGN_OR_RETURN(
      const TensorView logits_v,
      logits.View(DataType::kBFloat16, Shape({tokens, config.vocab_size})));

  INFERX_RETURN_IF_ERROR(gemm.LinearBF16(norm, lm_head, logits_v));

  INFERX_CUDA_RETURN_IF_ERROR(cudaDeviceSynchronize());

  return OkStatus();
}

StatusOr<Qwen2Model> Qwen2Model::Load(const ModelConfig& config,
                                      const Checkpoint& ckpt) {
  INFERX_RETURN_IF_ERROR(config.Validate());

  if (config.weight_dtype != DataType::kBFloat16) {
    return UnimplementedError(
        "M2 runs the stack in bf16; this checkpoint is ",
        DataTypeName(config.weight_dtype),
        ". Converting on upload is straightforward but is not done here, so "
        "that what runs is what the file contains.");
  }

  INFERX_ASSIGN_OR_RETURN(kernels::CublasLtGemm gemm,
                          kernels::CublasLtGemm::Create());

  auto impl = std::make_unique<Impl>(std::move(gemm));
  impl->config = config;

  const int64_t h = config.hidden_size;
  const int64_t inter = config.intermediate_size;

  INFERX_ASSIGN_OR_RETURN(
      const Tensor embed_host,
      ckpt.GetChecked("model.embed_tokens.weight",
                      Shape({config.vocab_size, h})));
  INFERX_ASSIGN_OR_RETURN(impl->embed,
                          Upload(embed_host, &impl->weight_buffers));

  // Tied embeddings: the output projection *is* the embedding matrix, so the
  // LM head is the same device buffer rather than a second 600 MB copy.
  if (config.tie_word_embeddings) {
    impl->lm_head = impl->embed;
  } else {
    INFERX_ASSIGN_OR_RETURN(
        const Tensor lm_host,
        ckpt.GetChecked("lm_head.weight", Shape({config.vocab_size, h})));
    INFERX_ASSIGN_OR_RETURN(impl->lm_head,
                            Upload(lm_host, &impl->weight_buffers));
  }

  INFERX_ASSIGN_OR_RETURN(const Tensor final_norm_host,
                          ckpt.GetChecked("model.norm.weight", Shape({h})));
  INFERX_ASSIGN_OR_RETURN(impl->final_norm,
                          Upload(final_norm_host, &impl->weight_buffers));

  impl->layers.reserve(static_cast<size_t>(config.num_hidden_layers));

  for (int64_t i = 0; i < config.num_hidden_layers; ++i) {
    const std::string p = absl::StrCat("model.layers.", i, ".");

    LayerWeights w;

    const auto load = [&](std::string_view suffix, const Shape& shape,
                          TensorView* dst) -> Status {
      INFERX_ASSIGN_OR_RETURN(const Tensor host,
                              ckpt.GetChecked(absl::StrCat(p, suffix), shape));
      INFERX_ASSIGN_OR_RETURN(*dst, Upload(host, &impl->weight_buffers));
      return OkStatus();
    };

    INFERX_RETURN_IF_ERROR(load("input_layernorm.weight", Shape({h}),
                                &w.input_norm));
    INFERX_RETURN_IF_ERROR(load("post_attention_layernorm.weight", Shape({h}),
                                &w.post_norm));

    INFERX_RETURN_IF_ERROR(load("self_attn.q_proj.weight",
                                Shape({config.q_dim(), h}), &w.q_w));
    INFERX_RETURN_IF_ERROR(load("self_attn.k_proj.weight",
                                Shape({config.kv_dim(), h}), &w.k_w));
    INFERX_RETURN_IF_ERROR(load("self_attn.v_proj.weight",
                                Shape({config.kv_dim(), h}), &w.v_w));
    INFERX_RETURN_IF_ERROR(load("self_attn.o_proj.weight",
                                Shape({h, config.q_dim()}), &w.o_w));

    if (config.attention_bias) {
      INFERX_RETURN_IF_ERROR(load("self_attn.q_proj.bias",
                                  Shape({config.q_dim()}), &w.q_b));
      INFERX_RETURN_IF_ERROR(load("self_attn.k_proj.bias",
                                  Shape({config.kv_dim()}), &w.k_b));
      INFERX_RETURN_IF_ERROR(load("self_attn.v_proj.bias",
                                  Shape({config.kv_dim()}), &w.v_b));
    }

    INFERX_RETURN_IF_ERROR(load("mlp.gate_proj.weight", Shape({inter, h}),
                                &w.gate_w));
    INFERX_RETURN_IF_ERROR(load("mlp.up_proj.weight", Shape({inter, h}),
                                &w.up_w));
    INFERX_RETURN_IF_ERROR(load("mlp.down_proj.weight", Shape({h, inter}),
                                &w.down_w));

    impl->layers.push_back(w);
  }

  for (const DeviceBuffer& b : impl->weight_buffers) {
    impl->weight_bytes += b.size();
  }

  return Qwen2Model(std::move(impl));
}

StatusOr<Qwen2Model> Qwen2Model::LoadFromDirectory(std::string_view dir) {
  INFERX_ASSIGN_OR_RETURN(const ModelConfig config,
                          ModelConfig::FromDirectory(dir));
  INFERX_ASSIGN_OR_RETURN(const Checkpoint ckpt, Checkpoint::Open(dir));

  return Load(config, ckpt);
}

Qwen2Model::Qwen2Model(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
Qwen2Model::~Qwen2Model() = default;
Qwen2Model::Qwen2Model(Qwen2Model&&) noexcept = default;
Qwen2Model& Qwen2Model::operator=(Qwen2Model&&) noexcept = default;

const ModelConfig& Qwen2Model::config() const { return impl_->config; }
size_t Qwen2Model::WeightBytes() const { return impl_->weight_bytes; }

namespace {

Status ValidateIds(const std::vector<int32_t>& ids, int64_t vocab) {
  if (ids.empty()) return InvalidArgumentError("token_ids is empty");

  for (size_t i = 0; i < ids.size(); ++i) {
    if (ids[i] < 0 || ids[i] >= vocab) {
      return InvalidArgumentError("token_ids[", i, "] = ", ids[i],
                                  " is outside the vocabulary [0, ", vocab,
                                  ")");
    }
  }

  return OkStatus();
}

/// Copies a bf16 logits row back to the host as fp32.
Status DownloadLogits(const DeviceBuffer& buf, int64_t offset_elems,
                      int64_t count, std::vector<float>* out) {
  std::vector<uint16_t> bits(static_cast<size_t>(count));

  INFERX_CUDA_RETURN_IF_ERROR(cudaMemcpy(
      bits.data(),
      static_cast<const std::byte*>(buf.data()) + offset_elems * 2,
      bits.size() * 2, cudaMemcpyDeviceToHost));

  out->resize(bits.size());

  for (size_t i = 0; i < bits.size(); ++i) {
    // bf16 is the top 16 bits of an fp32, so widening is a shift -- no lookup
    // table and no library call.
    const uint32_t word = static_cast<uint32_t>(bits[i]) << 16;
    float f;
    std::memcpy(&f, &word, sizeof(f));
    (*out)[i] = f;
  }

  return OkStatus();
}

}  // namespace

Status Qwen2Model::Forward(const std::vector<int32_t>& token_ids,
                           std::vector<float>* out_logits) {
  INFERX_RETURN_IF_ERROR(ValidateIds(token_ids, impl_->config.vocab_size));

  int64_t tokens = 0;
  INFERX_RETURN_IF_ERROR(impl_->RunForward(token_ids, &tokens));

  return DownloadLogits(impl_->logits.buf, 0,
                        tokens * impl_->config.vocab_size, out_logits);
}

Status Qwen2Model::AttachKvCache(int64_t num_blocks, int64_t block_size) {
  KvLayout layout;
  layout.entries_per_token = 2;  // K and V; MLA would say 1 (T11)
  layout.kv_heads = impl_->config.num_key_value_heads;
  layout.head_dim = impl_->config.head_dim;
  layout.dtype = DataType::kBFloat16;

  INFERX_ASSIGN_OR_RETURN(
      KvBlockPool pool,
      KvBlockPool::Create(impl_->config.num_hidden_layers, num_blocks,
                          block_size, layout));

  impl_->pool = std::make_unique<KvBlockPool>(std::move(pool));

  return OkStatus();
}

KvBlockPool* Qwen2Model::kv_pool() { return impl_->pool.get(); }

Status Qwen2Model::Step(const ForwardBatch& batch,
                        std::vector<float>* out_logits) {
  if (impl_->pool == nullptr) {
    return FailedPreconditionError(
        "Step requires a KV cache; call AttachKvCache first");
  }

  const int64_t total_slots =
      impl_->pool->num_blocks() * impl_->pool->block_size();

  INFERX_RETURN_IF_ERROR(
      batch.Validate(impl_->config.vocab_size, total_slots));

  INFERX_RETURN_IF_ERROR(impl_->RunPagedForward(batch));

  // Only the requested rows come back. Copied one at a time because the wanted
  // indices are arbitrary; at 151936 floats a row this is still far less
  // traffic than returning the whole thing.
  const int64_t vocab = impl_->config.vocab_size;

  out_logits->clear();
  out_logits->reserve(batch.logits_indices.size() *
                      static_cast<size_t>(vocab));

  for (const int32_t index : batch.logits_indices) {
    std::vector<float> row;
    INFERX_RETURN_IF_ERROR(
        DownloadLogits(impl_->logits.buf, index * vocab, vocab, &row));
    out_logits->insert(out_logits->end(), row.begin(), row.end());
  }

  return OkStatus();
}

Status Qwen2Model::ForwardLastLogits(const std::vector<int32_t>& token_ids,
                                     std::vector<float>* out_logits) {
  INFERX_RETURN_IF_ERROR(ValidateIds(token_ids, impl_->config.vocab_size));

  int64_t tokens = 0;
  INFERX_RETURN_IF_ERROR(impl_->RunForward(token_ids, &tokens));

  const int64_t vocab = impl_->config.vocab_size;

  return DownloadLogits(impl_->logits.buf, (tokens - 1) * vocab, vocab,
                        out_logits);
}

}  // namespace inferx::model
