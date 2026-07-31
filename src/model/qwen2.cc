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

#ifdef INFERX_WITH_FLASHINFER
#include "absl/types/span.h"
#include "inferx/kernels/flashinfer_attention.h"
#endif

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
  // Q, K and V concatenated along dim 0 into one [q_dim + 2*kv_dim, hidden]
  // tensor, and likewise gate and up into [2*intermediate, hidden]. Weights
  // concatenate along their output dimension, which is contiguous, so this is a
  // load-time arrangement rather than a runtime one -- and it turns six launches
  // per layer into two.
  TensorView qkv_w;
  TensorView qkv_b;  // undefined when the architecture has no attention bias
  TensorView o_w;
  TensorView post_norm;
  TensorView gate_up_w, down_w;
};

/// Uploads several host tensors end to end into one device buffer.
///
/// The concatenation happens here rather than on the device because it happens
/// once, at load, and a copy per tensor into the right offset is simpler than
/// any kernel that would do the same thing.
StatusOr<TensorView> UploadConcatenated(const std::vector<Tensor>& parts,
                                        std::vector<DeviceBuffer>* keep) {
  int64_t rows = 0;
  int64_t cols = parts.front().rank() == 2 ? parts.front().dim(1) : 0;
  size_t bytes = 0;

  for (const Tensor& t : parts) {
    rows += t.dim(0);
    bytes += static_cast<size_t>(t.nbytes());

    if (t.rank() == 2 && t.dim(1) != cols) {
      return InvalidArgumentError("cannot concatenate: widths ", cols, " and ",
                                  t.dim(1), " differ");
    }
  }

  INFERX_ASSIGN_OR_RETURN(DeviceBuffer buf,
                          DeviceBuffer::Allocate(bytes, DeviceId::Cuda(0)));

  size_t offset = 0;
  for (const Tensor& t : parts) {
    INFERX_CUDA_RETURN_IF_ERROR(
        cudaMemcpy(buf.data() + offset, t.data(),
                   static_cast<size_t>(t.nbytes()), cudaMemcpyHostToDevice));
    offset += static_cast<size_t>(t.nbytes());
  }

  keep->push_back(std::move(buf));

  const Shape shape = parts.front().rank() == 2 ? Shape({rows, cols})
                                                : Shape({rows});

  return TensorView::Create(keep->back().data(), parts.front().dtype(), shape,
                            DeviceId::Cuda(0));
}

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
  Scratch qkv_fused, gate_up_fused;
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

  // Everything device-side runs on this stream rather than the default one,
  // because a graph cannot be captured from the legacy default stream.
  cudaStream_t stream = nullptr;

#ifdef INFERX_WITH_FLASHINFER
  // The fast decode path. Absent when the submodule is not vendored, in which
  // case every batch takes the reference kernel and the engine still works.
  std::unique_ptr<kernels::FlashInferDecode> flashinfer;

  // CSR view of this step's block table, which is the form FlashInfer wants
  // (our own kernel reads the dense grid directly). Rebuilt per step.
  DeviceBuffer fi_indices_buf, fi_indptr_buf, fi_last_page_buf;
  int64_t fi_indices_capacity = 0;
  int64_t fi_seq_capacity = 0;
  std::vector<int32_t> fi_indptr_host;

  /// Set by PrepareBatchInputs: true when this batch is decode-shaped and the
  /// FlashInfer path is usable for it.
  bool fi_usable = false;
#endif

  /// Forced false while capturing. \see LaunchDecodeBody.
  bool allow_flashinfer = true;

  // One instantiated graph per decode shape. Keyed on (tokens, seqs, blocks):
  // a graph records fixed launch dimensions, so a batch of a different shape
  // needs its own. Decode shapes repeat forever, which is what makes this pay.
  struct DecodeGraph {
    int64_t tokens = 0;
    int64_t num_seqs = 0;
    int64_t max_blocks = 0;
    cudaGraphExec_t exec = nullptr;
  };
  std::vector<DecodeGraph> graphs;

  ~Impl() {
    for (DecodeGraph& g : graphs) {
      if (g.exec != nullptr) cudaGraphExecDestroy(g.exec);
    }
    if (stream != nullptr) cudaStreamDestroy(stream);
  }

  Status EnsureCapacity(int64_t tokens);
  Status RunForward(const std::vector<int32_t>& ids, int64_t* out_tokens);
  Status RunPagedForward(const ForwardBatch& batch);

  /// Host-side: sizes buffers and uploads this step's indices. Never captured.
  Status PrepareBatchInputs(const ForwardBatch& batch);

  /// Device-side: the entire forward pass, stream-ordered and capturable.
  Status LaunchDecodeBody(int64_t tokens, int64_t num_seqs,
                          int64_t max_blocks_per_seq);

  cudaGraphExec_t FindGraph(int64_t tokens, int64_t seqs, int64_t blocks) {
    for (DecodeGraph& g : graphs) {
      if (g.tokens == tokens && g.num_seqs == seqs && g.max_blocks == blocks) {
        return g.exec;
      }
    }
    return nullptr;
  }
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
  INFERX_RETURN_IF_ERROR(alloc(&qkv_fused, tokens * (qd + 2 * kvd), kBf16));
  INFERX_RETURN_IF_ERROR(alloc(&gate_up_fused, tokens * 2 * inter, kBf16));
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
      const TensorView qkv_v,
      qkv_fused.View(DataType::kBFloat16,
                     Shape({tokens, config.q_dim() + 2 * config.kv_dim()})));
  INFERX_ASSIGN_OR_RETURN(
      const TensorView gate_up_v,
      gate_up_fused.View(DataType::kBFloat16, Shape({tokens, 2 * inter})));

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

    INFERX_RETURN_IF_ERROR(gemm.LinearBF16(norm, layer.qkv_w, qkv_v));
    INFERX_RETURN_IF_ERROR(
        kernels::SplitQkvWithBias(qkv_v, layer.qkv_b, qv, kv_, vv));

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

    INFERX_RETURN_IF_ERROR(gemm.LinearBF16(norm, layer.gate_up_w, gate_up_v));
    INFERX_RETURN_IF_ERROR(kernels::SiluMulFused(gate_up_v, gate_v));
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

Status Qwen2Model::Impl::PrepareBatchInputs(const ForwardBatch& batch) {
  const int64_t tokens = batch.num_tokens();

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

#ifdef INFERX_WITH_FLASHINFER
  // FlashInfer's decode kernel serves exactly one query token per sequence.
  // A prefill, or a mixed batch, is a different shape entirely -- its ragged
  // prefill kernel is a separate integration -- so those keep the reference
  // kernel. Detecting it here rather than in the launch keeps the device half
  // free of host-side branching.
  fi_usable = flashinfer != nullptr && tokens == batch.num_seqs;

  if (fi_usable) {
    // Each sequence's cached length is its query position plus one, which is
    // also how many keys it will attend over.
    std::vector<int32_t> blocks_used(static_cast<size_t>(batch.num_seqs));
    std::vector<int32_t> last_page(static_cast<size_t>(batch.num_seqs));

    const int64_t block_size = pool->block_size();

    for (int64_t i = 0; i < tokens; ++i) {
      const int64_t seq = batch.seq_of_token[static_cast<size_t>(i)];
      const int64_t len = batch.positions[static_cast<size_t>(i)] + 1;
      const int64_t used = (len + block_size - 1) / block_size;

      blocks_used[static_cast<size_t>(seq)] = static_cast<int32_t>(used);

      // Never zero: a length that exactly fills its last page uses all of it.
      const int64_t rem = len - (used - 1) * block_size;
      last_page[static_cast<size_t>(seq)] = static_cast<int32_t>(rem);
    }

    std::vector<int32_t> indices;
    INFERX_RETURN_IF_ERROR(kernels::BuildCsrBlockTable(
        batch.block_table, batch.num_seqs, batch.max_blocks_per_seq,
        blocks_used, &indices, &fi_indptr_host));

    if (static_cast<int64_t>(indices.size()) > fi_indices_capacity) {
      INFERX_ASSIGN_OR_RETURN(
          fi_indices_buf,
          DeviceBuffer::Allocate(indices.size() * sizeof(int32_t),
                                 DeviceId::Cuda(0)));
      fi_indices_capacity = static_cast<int64_t>(indices.size());
    }

    if (batch.num_seqs > fi_seq_capacity) {
      INFERX_ASSIGN_OR_RETURN(
          fi_indptr_buf,
          DeviceBuffer::Allocate(
              static_cast<size_t>(batch.num_seqs + 1) * sizeof(int32_t),
              DeviceId::Cuda(0)));
      INFERX_ASSIGN_OR_RETURN(
          fi_last_page_buf,
          DeviceBuffer::Allocate(
              static_cast<size_t>(batch.num_seqs) * sizeof(int32_t),
              DeviceId::Cuda(0)));
      fi_seq_capacity = batch.num_seqs;
    }

    INFERX_RETURN_IF_ERROR(upload(fi_indices_buf, indices));
    INFERX_RETURN_IF_ERROR(upload(fi_indptr_buf, fi_indptr_host));
    INFERX_RETURN_IF_ERROR(upload(fi_last_page_buf, last_page));

    // Planned here, once per step, and deliberately not inside the launch: the
    // planner reads the indptr on the host and branches on it, which no graph
    // can record. Doing it here means all 36 layers share one plan *and* the
    // launch below becomes capturable.
    //
    // graph_safe unconditionally, even when no graph exists. It pins the
    // workspace offsets so a captured launch keeps addressing the right memory
    // as sequences grow, and using it always means the graphed and ungraphed
    // paths run identical kernels -- so a test that says "graphs do not change
    // the output" is testing dispatch rather than arithmetic.
    INFERX_RETURN_IF_ERROR(flashinfer->Plan(
        batch.num_seqs, config.num_attention_heads,
        config.num_key_value_heads, config.head_dim, block_size,
        absl::MakeConstSpan(fi_indptr_host), /*graph_safe=*/true, stream));
  }
#endif

  return OkStatus();
}

Status Qwen2Model::Impl::LaunchDecodeBody(int64_t tokens, int64_t num_seqs,
                                          int64_t max_blocks_per_seq) {
  const int64_t h = config.hidden_size;
  const int64_t heads = config.num_attention_heads;
  const int64_t kv_heads = config.num_key_value_heads;
  const int64_t hd = config.head_dim;
  const int64_t inter = config.intermediate_size;

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
          Shape({num_seqs, max_blocks_per_seq})));

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
      const TensorView qkv_v,
      qkv_fused.View(DataType::kBFloat16,
                     Shape({tokens, config.q_dim() + 2 * config.kv_dim()})));
  INFERX_ASSIGN_OR_RETURN(
      const TensorView gate_up_v,
      gate_up_fused.View(DataType::kBFloat16, Shape({tokens, 2 * inter})));

  const float attn_scale = 1.0f / std::sqrt(static_cast<float>(hd));

#ifdef INFERX_WITH_FLASHINFER
  // Capturable now. The planning that could not be recorded happens in
  // PrepareBatchInputs, once per step; what is left here is a kernel launch
  // whose every pointer is stable and whose every value lives in device memory
  // the plan rewrote. allow_flashinfer survives only so a caller can force the
  // reference kernel.
  const bool use_flashinfer = fi_usable && allow_flashinfer;
#endif

  INFERX_RETURN_IF_ERROR(kernels::EmbeddingLookup(embed, ids_v, x, stream));

  for (int64_t layer_index = 0;
       layer_index < static_cast<int64_t>(layers.size()); ++layer_index) {
    const LayerWeights& layer = layers[static_cast<size_t>(layer_index)];

    INFERX_CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(
        resid.Data(), x.Data(), static_cast<size_t>(x.NBytes()),
        cudaMemcpyDeviceToDevice, stream));

    INFERX_RETURN_IF_ERROR(kernels::RmsNorm(
        x, layer.input_norm, norm, static_cast<float>(config.rms_norm_eps), stream));

    // One GEMM and one split, where this used to be three GEMMs and three bias
    // adds. The split is unavoidable -- a fused row interleaves Q, K and V per
    // token, so Q is strided for more than one token -- but folding the bias
    // into it means the fused path still costs four launches fewer per layer.
    INFERX_RETURN_IF_ERROR(
        gemm.LinearBF16(norm, layer.qkv_w, qkv_v, stream));
    INFERX_RETURN_IF_ERROR(
        kernels::SplitQkvWithBias(qkv_v, layer.qkv_b, qv, kv_, vv, stream));

    INFERX_RETURN_IF_ERROR(kernels::RotaryEmbedding(
        q3, k3, pos_v, static_cast<float>(config.rope_theta), stream));

    // The cache is written *after* RoPE and *before* attention. Storing rotated
    // keys is what lets a cached key be reused at every later step without
    // re-rotating it -- the position it was rotated for is the position it
    // keeps.
    INFERX_ASSIGN_OR_RETURN(const TensorView k_cache,
                            pool->KeyCache(layer_index));
    INFERX_ASSIGN_OR_RETURN(const TensorView v_cache,
                            pool->ValueCache(layer_index));

    INFERX_RETURN_IF_ERROR(
        kernels::AppendToKvCache(k3, v3, k_cache, v_cache, slots_v, stream));

#ifdef INFERX_WITH_FLASHINFER
    if (use_flashinfer) {
      INFERX_ASSIGN_OR_RETURN(
          const TensorView fi_indices,
          TensorView::Create(fi_indices_buf.data(), DataType::kInt32,
                             Shape({static_cast<int64_t>(
                                 fi_indptr_host.back())}),
                             DeviceId::Cuda(0)));
      INFERX_ASSIGN_OR_RETURN(
          const TensorView fi_indptr,
          TensorView::Create(fi_indptr_buf.data(), DataType::kInt32,
                             Shape({num_seqs + 1}), DeviceId::Cuda(0)));
      INFERX_ASSIGN_OR_RETURN(
          const TensorView fi_last_page,
          TensorView::Create(fi_last_page_buf.data(), DataType::kInt32,
                             Shape({num_seqs}), DeviceId::Cuda(0)));

      INFERX_RETURN_IF_ERROR(flashinfer->Run(q3, k_cache, v_cache, fi_indices,
                                             fi_indptr, fi_last_page, a3,
                                             attn_scale, stream));
    } else
#endif
    {
      INFERX_RETURN_IF_ERROR(kernels::PagedAttention(
          q3, k_cache, v_cache, table_v, seq_v, pos_v, a3, attn_scale, stream));
    }

    INFERX_RETURN_IF_ERROR(gemm.LinearBF16(av, layer.o_w, x, stream));
    INFERX_RETURN_IF_ERROR(kernels::AddInPlace(x, resid, stream));

    INFERX_CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(
        resid.Data(), x.Data(), static_cast<size_t>(x.NBytes()),
        cudaMemcpyDeviceToDevice, stream));

    INFERX_RETURN_IF_ERROR(kernels::RmsNorm(
        x, layer.post_norm, norm, static_cast<float>(config.rms_norm_eps), stream));

    // Gate and up need no split: SiluMul is elementwise, so it reads both
    // halves straight out of the fused buffer. This fusion is free.
    INFERX_RETURN_IF_ERROR(
        gemm.LinearBF16(norm, layer.gate_up_w, gate_up_v, stream));
    INFERX_RETURN_IF_ERROR(
        kernels::SiluMulFused(gate_up_v, gate_v, stream));
    INFERX_RETURN_IF_ERROR(gemm.LinearBF16(gate_v, layer.down_w, x, stream));
    INFERX_RETURN_IF_ERROR(kernels::AddInPlace(x, resid, stream));
  }

  INFERX_RETURN_IF_ERROR(kernels::RmsNorm(
      x, final_norm, norm, static_cast<float>(config.rms_norm_eps), stream));

  INFERX_ASSIGN_OR_RETURN(
      const TensorView logits_v,
      logits.View(DataType::kBFloat16, Shape({tokens, config.vocab_size})));

  INFERX_RETURN_IF_ERROR(gemm.LinearBF16(norm, lm_head, logits_v, stream));

  return OkStatus();
}


Status Qwen2Model::Impl::RunPagedForward(const ForwardBatch& batch) {
  INFERX_RETURN_IF_ERROR(PrepareBatchInputs(batch));

  const int64_t tokens = batch.num_tokens();

  cudaGraphExec_t graph =
      FindGraph(tokens, batch.num_seqs, batch.max_blocks_per_seq);

  if (graph != nullptr) {
    // The replay reads exactly the buffers PrepareBatchInputs just wrote. That
    // is the whole contract: a graph fixes the *structure* of the step -- which
    // kernels, what dimensions, which pointers -- while every value it reads
    // still comes from memory that was updated a moment ago. Sequence lengths,
    // block tables and token ids all change freely between replays; only the
    // shape may not.
    INFERX_CUDA_RETURN_IF_ERROR(cudaGraphLaunch(graph, stream));
  } else {
    INFERX_RETURN_IF_ERROR(LaunchDecodeBody(tokens, batch.num_seqs,
                                            batch.max_blocks_per_seq));
  }

  INFERX_CUDA_RETURN_IF_ERROR(cudaStreamSynchronize(stream));

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

  // A dedicated stream, non-blocking. Two reasons, and the first is not
  // optional: the legacy default stream cannot be captured at all, and
  // attempting it returns cudaErrorStreamCaptureUnsupported. Non-blocking so it
  // does not implicitly synchronize against the default stream, which would
  // reintroduce exactly the serialization graphs are here to remove.
  INFERX_CUDA_RETURN_IF_ERROR(
      cudaStreamCreateWithFlags(&impl->stream, cudaStreamNonBlocking));

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

    // Fetch the three projections and upload them as one tensor. They are
    // concatenated along dim 0, which is contiguous, so the fused weight is
    // byte-identical to the three laid end to end.
    {
      std::vector<Tensor> qkv;
      for (const auto& [name, rows] :
           {std::pair<std::string_view, int64_t>{"self_attn.q_proj.weight",
                                                 config.q_dim()},
            {"self_attn.k_proj.weight", config.kv_dim()},
            {"self_attn.v_proj.weight", config.kv_dim()}}) {
        INFERX_ASSIGN_OR_RETURN(
            Tensor t, ckpt.GetChecked(absl::StrCat(p, name), Shape({rows, h})));
        qkv.push_back(std::move(t));
      }
      INFERX_ASSIGN_OR_RETURN(w.qkv_w,
                              UploadConcatenated(qkv, &impl->weight_buffers));
    }

    if (config.attention_bias) {
      std::vector<Tensor> qkv_bias;
      for (const auto& [name, rows] :
           {std::pair<std::string_view, int64_t>{"self_attn.q_proj.bias",
                                                 config.q_dim()},
            {"self_attn.k_proj.bias", config.kv_dim()},
            {"self_attn.v_proj.bias", config.kv_dim()}}) {
        INFERX_ASSIGN_OR_RETURN(
            Tensor t, ckpt.GetChecked(absl::StrCat(p, name), Shape({rows})));
        qkv_bias.push_back(std::move(t));
      }
      INFERX_ASSIGN_OR_RETURN(
          w.qkv_b, UploadConcatenated(qkv_bias, &impl->weight_buffers));
    }

    INFERX_RETURN_IF_ERROR(load("self_attn.o_proj.weight",
                                Shape({h, config.q_dim()}), &w.o_w));

    {
      std::vector<Tensor> gate_up;
      for (const std::string_view name :
           {"mlp.gate_proj.weight", "mlp.up_proj.weight"}) {
        INFERX_ASSIGN_OR_RETURN(
            Tensor t,
            ckpt.GetChecked(absl::StrCat(p, name), Shape({inter, h})));
        gate_up.push_back(std::move(t));
      }
      INFERX_ASSIGN_OR_RETURN(w.gate_up_w,
                              UploadConcatenated(gate_up,
                                                 &impl->weight_buffers));
    }

    INFERX_RETURN_IF_ERROR(load("mlp.down_proj.weight", Shape({h, inter}),
                                &w.down_w));

    impl->layers.push_back(w);
  }

#ifdef INFERX_WITH_FLASHINFER
  INFERX_ASSIGN_OR_RETURN(kernels::FlashInferDecode fi,
                          kernels::FlashInferDecode::Create());
  impl->flashinfer =
      std::make_unique<kernels::FlashInferDecode>(std::move(fi));
#endif

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


Status Qwen2Model::CaptureDecodeGraph(int64_t num_seqs,
                                      int64_t max_blocks_per_seq) {
  if (impl_->pool == nullptr) {
    return FailedPreconditionError(
        "CaptureDecodeGraph requires a KV cache; call AttachKvCache first");
  }

  if (num_seqs <= 0 || max_blocks_per_seq <= 0) {
    return InvalidArgumentError("graph shape must be positive, got num_seqs=",
                                num_seqs, " max_blocks_per_seq=",
                                max_blocks_per_seq);
  }

  // Decode is one token per sequence, which is the only shape worth capturing:
  // prefill lengths vary per request and would need a graph each.
  const int64_t tokens = num_seqs;

  if (impl_->FindGraph(tokens, num_seqs, max_blocks_per_seq) != nullptr) {
    return OkStatus();
  }

  // A dummy batch of the right shape, so every buffer is allocated and every
  // pointer settled before capture begins. Capture records addresses; anything
  // allocated during it would be recorded and then freed.
  ForwardBatch probe;
  probe.num_seqs = num_seqs;
  probe.max_blocks_per_seq = max_blocks_per_seq;
  probe.block_table.assign(
      static_cast<size_t>(num_seqs * max_blocks_per_seq), 0);

  for (int64_t s = 0; s < num_seqs; ++s) {
    probe.token_ids.push_back(0);
    probe.positions.push_back(0);
    probe.seq_of_token.push_back(static_cast<int32_t>(s));
    probe.slots.push_back(0);
  }
  probe.logits_indices.push_back(0);

  INFERX_RETURN_IF_ERROR(impl_->PrepareBatchInputs(probe));

  // A warm-up run outside capture. cuBLASLt picks and caches an algorithm on
  // first use for a shape, and that selection does device work of its own; if
  // it happened during capture it would be baked into the graph.
  INFERX_RETURN_IF_ERROR(
      impl_->LaunchDecodeBody(tokens, num_seqs, max_blocks_per_seq));
  INFERX_CUDA_RETURN_IF_ERROR(cudaStreamSynchronize(impl_->stream));

  INFERX_CUDA_RETURN_IF_ERROR(cudaStreamBeginCapture(
      impl_->stream, cudaStreamCaptureModeThreadLocal));

  const Status body =
      impl_->LaunchDecodeBody(tokens, num_seqs, max_blocks_per_seq);

  cudaGraph_t graph = nullptr;
  const cudaError_t ended = cudaStreamEndCapture(impl_->stream, &graph);

  // Capture must be ended even if the body failed, or the stream stays in
  // capture mode and every later launch on it fails with an opaque error.
  INFERX_RETURN_IF_ERROR(body);
  INFERX_CUDA_RETURN_IF_ERROR(ended);

  cudaGraphExec_t exec = nullptr;
  const cudaError_t instantiated =
      cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0);

  cudaGraphDestroy(graph);  // the executable owns what it needs
  INFERX_CUDA_RETURN_IF_ERROR(instantiated);

  impl_->graphs.push_back({tokens, num_seqs, max_blocks_per_seq, exec});

  return OkStatus();
}

int64_t Qwen2Model::captured_graphs() const {
  return static_cast<int64_t>(impl_->graphs.size());
}

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
