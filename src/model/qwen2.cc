#include "inferx/model/qwen2.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
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
#include "inferx/kernels/quantize.h"

#ifdef INFERX_WITH_FLASHINFER
#include "absl/types/span.h"
#include "inferx/kernels/flashinfer_attention.h"
#include "inferx/kernels/flashinfer_prefill.h"
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

  // Populated by QuantizeWeightsToF8. Each carries a device-resident dequant
  // scale, which cuBLASLt folds into the epilogue.
  TensorView qkv_w8, o_w8, gate_up_w8, down_w8;
  // Indices into Impl::weight_buffers for the four large tensors, so FP8
  // conversion can release exactly those. The norm weights and the embedding
  // share that vector and must survive.
  int qkv_buf = -1, o_buf = -1, gate_up_buf = -1, down_buf = -1;
  float* qkv_s = nullptr;
  float* o_s = nullptr;
  float* gate_up_s = nullptr;
  float* down_s = nullptr;
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

  // Page-locked staging for the per-step index arrays, and for the logits on
  // the way back.
  //
  // A cudaMemcpy from ordinary pageable memory is synchronous *and* stages
  // through a driver-owned bounce buffer, so the eight tiny index uploads a
  // step needs cost 0.149 ms of host time between them -- far more than the
  // bytes justify. From pinned memory the same copies are asynchronous enqueues
  // of a few microseconds each. Measured in bench/decode_breakdown_bench.cc.
  void* pinned = nullptr;
  size_t pinned_bytes = 0;
  void* pinned_logits = nullptr;
  size_t pinned_logits_bytes = 0;

  // Events bracketing the device portion of a step, so the split between work
  // and the host wrapper around it can be measured rather than inferred.
  cudaEvent_t step_begin = nullptr;
  cudaEvent_t step_end = nullptr;
  float last_device_ms = 0.0f;

  /// Grows the index staging buffer. Contents are not preserved; every step
  /// rewrites all of it.
  Status EnsurePinned(size_t bytes) {
    if (bytes <= pinned_bytes) return OkStatus();

    if (pinned != nullptr) cudaFreeHost(pinned);
    pinned = nullptr;
    pinned_bytes = 0;

    INFERX_CUDA_RETURN_IF_ERROR(
        cudaHostAlloc(&pinned, bytes, cudaHostAllocDefault));
    pinned_bytes = bytes;

    return OkStatus();
  }

#ifdef INFERX_WITH_FLASHINFER
  // The fast decode path. Absent when the submodule is not vendored, in which
  // case every batch takes the reference kernel and the engine still works.
  std::unique_ptr<kernels::FlashInferDecode> flashinfer;
  std::unique_ptr<kernels::FlashInferPrefill> flashinfer_prefill;

  // CSR view of this step's block table, which is the form FlashInfer wants
  // (our own kernel reads the dense grid directly). Rebuilt per step.
  DeviceBuffer fi_indices_buf, fi_indptr_buf, fi_last_page_buf;
  DeviceBuffer fi_qo_indptr_buf;
  int64_t fi_indices_capacity = 0;
  int64_t fi_seq_capacity = 0;
  std::vector<int32_t> fi_indptr_host;
  std::vector<int32_t> fi_qo_indptr_host;

  /// Set by PrepareBatchInputs: true when this batch is decode-shaped and the
  /// FlashInfer path is usable for it.
  bool fi_usable = false;
  bool fi_prefill_usable = false;

#endif

  /// Longest context in the batch PrepareBatchInputs last saw, which sizes the
  /// reference attention kernel's shared-memory tile. Outside the FlashInfer
  /// guard because the reference kernel is exactly what runs without it.
  int64_t batch_max_context = 0;

  /// Forced false while capturing. \see LaunchDecodeBody.
  bool allow_flashinfer = true;

  // FP8 weight mode. The activation staging buffers are sized for the widest
  // GEMM input a layer has, which is the intermediate width feeding down_proj.
  // On-device sampling, and the buffers that let a decode step feed its own
  // successor without the host in between.
  bool sample_on_device = false;
  DeviceBuffer sampled_ids;      // [logits rows] int32, this step's tokens
  DeviceBuffer sample_slots;     // where each sampled token lands in token_ids
  DeviceBuffer sample_rows;      // which logits rows to sample
  void* pinned_sampled = nullptr;  // host-visible copy, read one step late
  cudaEvent_t sampled_ready = nullptr;
  int64_t sampled_count = 0;

  bool weights_f8 = false;
  std::vector<DeviceBuffer> f8_buffers;
  DeviceBuffer act_f8;        // quantized activations, reused every GEMM
  DeviceBuffer act_scales;    // one float per GEMM input per layer

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
    if (sampled_ready != nullptr) cudaEventDestroy(sampled_ready);
    if (pinned_sampled != nullptr) cudaFreeHost(pinned_sampled);
    if (step_begin != nullptr) cudaEventDestroy(step_begin);
    if (step_end != nullptr) cudaEventDestroy(step_end);
    if (pinned != nullptr) cudaFreeHost(pinned);
    if (pinned_logits != nullptr) cudaFreeHost(pinned_logits);
    if (stream != nullptr) cudaStreamDestroy(stream);
  }

  /// One projection, in whichever precision the model is configured for.
  ///
  /// The FP8 form quantizes its input first, dynamically, one scale per tensor.
  /// That costs a launch per GEMM -- against roughly half the weight bytes
  /// saved, which at these sizes is the trade that pays.
  Status Linear(const TensorView& in, const TensorView& w_bf16,
                const TensorView& w_f8, const float* w_scale,
                const TensorView& out, int scale_slot) {
    if (!weights_f8 || !w_f8.IsDefined()) {
      return gemm.LinearBF16(in, w_bf16, out, stream);
    }

    auto* act_scale = reinterpret_cast<float*>(act_scales.data()) + scale_slot;

    INFERX_ASSIGN_OR_RETURN(
        const TensorView in_f8,
        TensorView::Create(act_f8.data(), DataType::kFloat8E4M3FN,
                           in.GetShape(), DeviceId::Cuda(0)));

    INFERX_RETURN_IF_ERROR(
        kernels::QuantizeToF8E4M3Dynamic(in, in_f8, act_scale, stream));

    return gemm.LinearF8E4M3(in_f8, w_f8, out, act_scale, w_scale, stream);
  }

  /// Grows the quantized-activation staging to cover `tokens`.
  ///
  /// Separate from EnsureCapacity's early return, and called before it, because
  /// the two do not grow together: quantization can be switched on when no
  /// forward has run at all, at which point capacity_tokens is zero and sizing
  /// this from it yields a null buffer that the first step then writes to. That
  /// is exactly the bug this shape prevents.
  Status EnsureActivationF8(int64_t tokens) {
    if (!weights_f8 || tokens <= 0) return OkStatus();

    const int64_t widest = std::max({config.hidden_size,
                                     config.q_dim() + 2 * config.kv_dim(),
                                     config.intermediate_size});
    const size_t need = static_cast<size_t>(tokens * widest);

    if (need <= act_f8.size()) return OkStatus();

    INFERX_ASSIGN_OR_RETURN(act_f8,
                            DeviceBuffer::Allocate(need, DeviceId::Cuda(0)));
    return OkStatus();
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
  INFERX_RETURN_IF_ERROR(EnsureActivationF8(tokens));

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

  // Sized for the largest download any caller can ask for: every token's row.
  const size_t logits_bytes = static_cast<size_t>(tokens) *
                              static_cast<size_t>(vocab) * 2;

  if (logits_bytes > pinned_logits_bytes) {
    if (pinned_logits != nullptr) cudaFreeHost(pinned_logits);
    pinned_logits = nullptr;
    pinned_logits_bytes = 0;

    INFERX_CUDA_RETURN_IF_ERROR(
        cudaHostAlloc(&pinned_logits, logits_bytes, cudaHostAllocDefault));
    pinned_logits_bytes = logits_bytes;
  }

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

  // Reserved before anything is staged, and generously: four per-token arrays,
  // a dense block table, and the CSR form of that table plus its two small
  // companions. Sizing once is what makes the buffer safe to hold pointers into
  // while copies are in flight.
  {
    const size_t per_token = 4 * static_cast<size_t>(tokens);
    const size_t table = static_cast<size_t>(batch.num_seqs) *
                         static_cast<size_t>(batch.max_blocks_per_seq);
    const size_t needed =
        (per_token + 2 * table + 3 * static_cast<size_t>(batch.num_seqs) + 2) *
        sizeof(int32_t);

    INFERX_RETURN_IF_ERROR(EnsurePinned(needed));
  }

  // Every array a step uploads, staged end to end through one pinned buffer.
  // The offset walks forward across the calls below; nothing overlaps, because
  // all of it has to be resident until the copies complete.
  size_t staged = 0;

  const auto upload = [&](const DeviceBuffer& buf,
                          const std::vector<int32_t>& src) -> Status {
    const size_t bytes = src.size() * sizeof(int32_t);

    // Deliberately an error rather than a growth. Reallocating here would free
    // the buffer that the copies already enqueued are still reading from -- an
    // asynchronous use-after-free that would corrupt indices intermittently and
    // be nearly impossible to attribute. The buffer is sized once, before any
    // copy, by the reservation above.
    if (staged + bytes > pinned_bytes) {
      return InternalError("pinned staging exhausted: needed ", staged + bytes,
                           " bytes but reserved ", pinned_bytes);
    }

    std::memcpy(static_cast<std::byte*>(pinned) + staged, src.data(), bytes);

    INFERX_CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(
        buf.data(), static_cast<std::byte*>(pinned) + staged, bytes,
        cudaMemcpyHostToDevice, stream));

    staged += bytes;
    return OkStatus();
  };

  // Skipped when the sampler already wrote them: overwriting would replace the
  // token the GPU chose with whatever placeholder the caller passed.
  if (!batch.tokens_from_device) {
    INFERX_RETURN_IF_ERROR(upload(token_ids, batch.token_ids));
  }
  INFERX_RETURN_IF_ERROR(upload(positions, batch.positions));
  INFERX_RETURN_IF_ERROR(upload(slots_buf, batch.slots));
  INFERX_RETURN_IF_ERROR(upload(seq_of_token_buf, batch.seq_of_token));
  INFERX_RETURN_IF_ERROR(upload(block_table_buf, batch.block_table));

  // The longest key sequence any query in this batch attends over. Host-side
  // because `positions` is host-side here; the device never needs it. Computed
  // unconditionally: it sizes the reference attention kernel, which is the one
  // that runs when FlashInfer is absent or when the batch is a prefill.
  batch_max_context = 0;
  for (const int32_t p : batch.positions) {
    batch_max_context = std::max<int64_t>(batch_max_context, p + 1);
  }

#ifdef INFERX_WITH_FLASHINFER
  // Decode serves one query per sequence; the ragged prefill kernel serves
  // every other all-contributing shape.  A deferred sequence has no query rows
  // and no length in ForwardBatch, so a mixed step containing one still uses
  // the reference kernel until M5 makes sequence lengths explicit in the plan.
  fi_usable = false;
  fi_prefill_usable = false;

  if (flashinfer != nullptr || flashinfer_prefill != nullptr) {
    // Each sequence's cached length is its query position plus one, which is
    // also how many keys it will attend over.
    std::vector<int32_t> blocks_used(static_cast<size_t>(batch.num_seqs));
    std::vector<int32_t> last_page(static_cast<size_t>(batch.num_seqs));

    const int64_t block_size = pool->block_size();

    std::vector<int32_t> query_counts(static_cast<size_t>(batch.num_seqs));
    std::vector<int32_t> previous_position(static_cast<size_t>(batch.num_seqs),
                                           -1);
    int64_t previous_seq = -1;

    for (int64_t i = 0; i < tokens; ++i) {
      const int64_t seq = batch.seq_of_token[static_cast<size_t>(i)];
      const int64_t len = batch.positions[static_cast<size_t>(i)] + 1;
      const int64_t used = (len + block_size - 1) / block_size;

      if (seq < previous_seq ||
          (previous_position[static_cast<size_t>(seq)] >= 0 &&
           batch.positions[static_cast<size_t>(i)] !=
               previous_position[static_cast<size_t>(seq)] + 1)) {
        return InvalidArgumentError(
            "FlashInfer prefill requires rows grouped by sequence with "
            "contiguous positions");
      }
      previous_seq = seq;
      previous_position[static_cast<size_t>(seq)] =
          batch.positions[static_cast<size_t>(i)];

      ++query_counts[static_cast<size_t>(seq)];
      blocks_used[static_cast<size_t>(seq)] = static_cast<int32_t>(used);

      // Never zero: a length that exactly fills its last page uses all of it.
      const int64_t rem = len - (used - 1) * block_size;
      last_page[static_cast<size_t>(seq)] = static_cast<int32_t>(rem);
    }

    bool every_sequence_contributes = true;
    bool exactly_one_query_per_sequence = true;
    fi_qo_indptr_host.assign(static_cast<size_t>(batch.num_seqs + 1), 0);
    for (int64_t seq = 0; seq < batch.num_seqs; ++seq) {
      const int32_t count = query_counts[static_cast<size_t>(seq)];
      every_sequence_contributes &= count > 0;
      exactly_one_query_per_sequence &= count == 1;
      fi_qo_indptr_host[static_cast<size_t>(seq + 1)] =
          fi_qo_indptr_host[static_cast<size_t>(seq)] + count;

    }

    fi_usable = flashinfer != nullptr && exactly_one_query_per_sequence;
    // Wrapper output, immediate logits, and every cached per-layer K/V element
    // are repeatable, but long graph-served continuations still diverge
    // intermittently unless CUDA launches are serialized. That points at a
    // later timing race, so keep multi-request prefill on the reference path.
    // The environment override lets regression tests exercise the quarantined
    // route without making it the serving default.
    fi_prefill_usable = flashinfer_prefill != nullptr && !fi_usable &&
                        (batch.num_seqs == 1 ||
                         std::getenv("INFERX_EXPERIMENTAL_RAGGED_PREFILL") !=
                             nullptr) &&
                        every_sequence_contributes;

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
      INFERX_ASSIGN_OR_RETURN(
          fi_qo_indptr_buf,
          DeviceBuffer::Allocate(
              static_cast<size_t>(batch.num_seqs + 1) * sizeof(int32_t),
              DeviceId::Cuda(0)));
      fi_seq_capacity = batch.num_seqs;
    }

    INFERX_RETURN_IF_ERROR(upload(fi_indices_buf, indices));
    INFERX_RETURN_IF_ERROR(upload(fi_indptr_buf, fi_indptr_host));
    INFERX_RETURN_IF_ERROR(upload(fi_last_page_buf, last_page));
    INFERX_RETURN_IF_ERROR(upload(fi_qo_indptr_buf, fi_qo_indptr_host));

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
    if (fi_usable) {
      INFERX_RETURN_IF_ERROR(flashinfer->Plan(
          batch.num_seqs, config.num_attention_heads,
          config.num_key_value_heads, config.head_dim, block_size,
          absl::MakeConstSpan(fi_indptr_host), /*graph_safe=*/true, stream));
    } else if (fi_prefill_usable) {
      INFERX_RETURN_IF_ERROR(flashinfer_prefill->Plan(
          batch.num_seqs, config.num_attention_heads,
          config.num_key_value_heads, config.head_dim, block_size,
          absl::MakeConstSpan(fi_qo_indptr_host),
          absl::MakeConstSpan(fi_indptr_host), tokens, stream));
    }
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
  const bool use_flashinfer_prefill = fi_prefill_usable && allow_flashinfer;
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
    INFERX_RETURN_IF_ERROR(Linear(norm, layer.qkv_w, layer.qkv_w8,
                                  layer.qkv_s, qkv_v, 0));
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
    if (use_flashinfer || use_flashinfer_prefill) {
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

      if (use_flashinfer) {
        INFERX_RETURN_IF_ERROR(flashinfer->Run(
            q3, k_cache, v_cache, fi_indices, fi_indptr, fi_last_page, a3,
            attn_scale, stream));
      } else {
        INFERX_ASSIGN_OR_RETURN(
            const TensorView fi_qo_indptr,
            TensorView::Create(fi_qo_indptr_buf.data(), DataType::kInt32,
                               Shape({num_seqs + 1}), DeviceId::Cuda(0)));
        INFERX_RETURN_IF_ERROR(flashinfer_prefill->Run(
            q3, k_cache, v_cache, fi_qo_indptr, fi_indices, fi_indptr,
            fi_last_page, a3, attn_scale, stream));
      }
    } else
#endif
    {
      // The batch's own longest context, not the block table's width. See
      // PagedAttention's contract: sizing the tile from the table made a short
      // prompt demand shared memory proportional to max_seq_len.
      INFERX_RETURN_IF_ERROR(kernels::PagedAttention(
          q3, k_cache, v_cache, table_v, seq_v, pos_v, a3, attn_scale,
          batch_max_context, stream));
    }

    INFERX_RETURN_IF_ERROR(Linear(av, layer.o_w, layer.o_w8, layer.o_s, x, 1));
    INFERX_RETURN_IF_ERROR(kernels::AddInPlace(x, resid, stream));

    INFERX_CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(
        resid.Data(), x.Data(), static_cast<size_t>(x.NBytes()),
        cudaMemcpyDeviceToDevice, stream));

    INFERX_RETURN_IF_ERROR(kernels::RmsNorm(
        x, layer.post_norm, norm, static_cast<float>(config.rms_norm_eps), stream));

    // Gate and up need no split: SiluMul is elementwise, so it reads both
    // halves straight out of the fused buffer. This fusion is free.
    INFERX_RETURN_IF_ERROR(Linear(norm, layer.gate_up_w, layer.gate_up_w8,
                                  layer.gate_up_s, gate_up_v, 2));
    INFERX_RETURN_IF_ERROR(
        kernels::SiluMulFused(gate_up_v, gate_v, stream));
    INFERX_RETURN_IF_ERROR(Linear(gate_v, layer.down_w, layer.down_w8,
                                  layer.down_s, x, 3));
    INFERX_RETURN_IF_ERROR(kernels::AddInPlace(x, resid, stream));
  }

  INFERX_RETURN_IF_ERROR(kernels::RmsNorm(
      x, final_norm, norm, static_cast<float>(config.rms_norm_eps), stream));

  INFERX_ASSIGN_OR_RETURN(
      const TensorView logits_v,
      logits.View(DataType::kBFloat16, Shape({tokens, config.vocab_size})));

  INFERX_RETURN_IF_ERROR(gemm.LinearBF16(norm, lm_head, logits_v, stream));

  // Sampling closes the loop inside the captured region. The argmax writes each
  // sequence's next token, and the scatter drops it straight into the token
  // buffer the *next* replay will read. Nothing crosses to the host on this
  // path, which is what lets the host run ahead: it can compute positions and
  // slots for step N+1 -- both predictable -- without knowing what step N
  // produced.
  //
  // Only the requested logits rows are sampled, which for decode is one per
  // sequence.
  if (sample_on_device && sampled_count > 0) {
    INFERX_ASSIGN_OR_RETURN(
        const TensorView wanted,
        logits.View(DataType::kBFloat16,
                    Shape({tokens, config.vocab_size})));
    INFERX_ASSIGN_OR_RETURN(
        const TensorView ids_out,
        TensorView::Create(sampled_ids.data(), DataType::kInt32,
                           Shape({sampled_count}), DeviceId::Cuda(0)));
    INFERX_ASSIGN_OR_RETURN(
        const TensorView slots_out,
        TensorView::Create(sample_slots.data(), DataType::kInt32,
                           Shape({sampled_count}), DeviceId::Cuda(0)));

    INFERX_ASSIGN_OR_RETURN(
        const TensorView rows_in,
        TensorView::Create(sample_rows.data(), DataType::kInt32,
                           Shape({sampled_count}), DeviceId::Cuda(0)));

    INFERX_RETURN_IF_ERROR(
        kernels::ArgmaxSample(wanted, rows_in, ids_out, stream));

    INFERX_ASSIGN_OR_RETURN(
        const TensorView next_ids,
        TensorView::Create(token_ids.data(), DataType::kInt32, Shape({tokens}),
                           DeviceId::Cuda(0)));

    INFERX_RETURN_IF_ERROR(
        kernels::ScatterTokens(ids_out, next_ids, slots_out, stream));

    // A single async copy out, not waited on here. The host reads it a step
    // later, which is §4 step 8: results from step N are consumed at the start
    // of step N+1.
    // The copy out is stream-ordered work and belongs in the graph. The event
    // that tells the host it has landed does NOT: an event recorded inside a
    // captured region becomes a graph-internal dependency node, and replaying
    // the graph never signals the host-visible event. cudaEventSynchronize then
    // returns immediately on something that was never set, and generation
    // reports thousands of tokens a second while reading a stale buffer. The
    // record lives in StepAsync, after the launch.
    INFERX_CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(
        pinned_sampled, sampled_ids.data(),
        static_cast<size_t>(sampled_count) * sizeof(int32_t),
        cudaMemcpyDeviceToHost, stream));
  }

  return OkStatus();
}


Status Qwen2Model::Impl::RunPagedForward(const ForwardBatch& batch) {
  INFERX_RETURN_IF_ERROR(PrepareBatchInputs(batch));

  const int64_t tokens = batch.num_tokens();

  cudaGraphExec_t graph =
      FindGraph(tokens, batch.num_seqs, batch.max_blocks_per_seq);

  INFERX_CUDA_RETURN_IF_ERROR(cudaEventRecord(step_begin, stream));

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

  INFERX_CUDA_RETURN_IF_ERROR(cudaEventRecord(step_end, stream));
  INFERX_CUDA_RETURN_IF_ERROR(cudaStreamSynchronize(stream));
  INFERX_CUDA_RETURN_IF_ERROR(
      cudaEventElapsedTime(&last_device_ms, step_begin, step_end));

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
  INFERX_CUDA_RETURN_IF_ERROR(cudaEventCreate(&impl->step_begin));
  INFERX_CUDA_RETURN_IF_ERROR(cudaEventCreate(&impl->step_end));

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
      w.qkv_buf = static_cast<int>(impl->weight_buffers.size()) - 1;
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
    w.o_buf = static_cast<int>(impl->weight_buffers.size()) - 1;

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
      w.gate_up_buf = static_cast<int>(impl->weight_buffers.size()) - 1;
    }

    INFERX_RETURN_IF_ERROR(load("mlp.down_proj.weight", Shape({h, inter}),
                                &w.down_w));
    w.down_buf = static_cast<int>(impl->weight_buffers.size()) - 1;

    impl->layers.push_back(w);
  }

#ifdef INFERX_WITH_FLASHINFER
  INFERX_ASSIGN_OR_RETURN(kernels::FlashInferDecode fi,
                          kernels::FlashInferDecode::Create());
  impl->flashinfer =
      std::make_unique<kernels::FlashInferDecode>(std::move(fi));
  INFERX_ASSIGN_OR_RETURN(impl->flashinfer_prefill,
                          kernels::FlashInferPrefill::Create());
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

/// Copies a bf16 logits row back to the host as fp32, through pinned memory.
///
/// A vocabulary row is 304 KB, and from pageable memory that copy costs 0.054 ms
/// of host time per step. Pinned, it is a stream-ordered transfer like any
/// other.
Status DownloadLogitsPinned(const DeviceBuffer& buf, int64_t offset_elems,
                            int64_t count, void* pinned, cudaStream_t stream,
                            std::vector<float>* out) {
  const auto* bits = static_cast<const uint16_t*>(pinned);

  INFERX_CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(
      pinned, static_cast<const std::byte*>(buf.data()) + offset_elems * 2,
      static_cast<size_t>(count) * 2, cudaMemcpyDeviceToHost, stream));
  INFERX_CUDA_RETURN_IF_ERROR(cudaStreamSynchronize(stream));

  out->resize(static_cast<size_t>(count));

  for (size_t i = 0; i < static_cast<size_t>(count); ++i) {
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
  INFERX_RETURN_IF_ERROR(RequireBf16Weights());
  INFERX_RETURN_IF_ERROR(ValidateIds(token_ids, impl_->config.vocab_size));

  int64_t tokens = 0;
  INFERX_RETURN_IF_ERROR(impl_->RunForward(token_ids, &tokens));

  return DownloadLogitsPinned(impl_->logits.buf, 0,
                              tokens * impl_->config.vocab_size,
                              impl_->pinned_logits, impl_->stream, out_logits);
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


Status Qwen2Model::ReserveActivations(int64_t max_tokens) {
  if (max_tokens <= 0) {
    return InvalidArgumentError("max_tokens must be positive, got ",
                                max_tokens);
  }

  return impl_->EnsureCapacity(max_tokens);
}

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
  // A block of its own to scribble in.
  //
  // The probe runs the real decode body, which means it really appends keys and
  // values to the KV cache. Left at slot 0 -- as it was -- that write lands in
  // physical block 0, and the free list hands block 0 out first, so capturing a
  // graph overwrote the *first live sequence's* position-0 keys and values in
  // every layer. That was R9: not a graph bug at all, since a step run
  // launch-by-launch after an unrelated capture was corrupted just the same.
  //
  // Borrowed and returned rather than reserved forever: any sequence that later
  // receives this block writes its own keys and values over the probe's before
  // reading them back.
  INFERX_ASSIGN_OR_RETURN(const int32_t scratch_block,
                          impl_->pool->AllocateBlock());

  struct ScratchGuard {
    KvBlockPool* pool;
    int32_t block;
    ~ScratchGuard() { (void)pool->FreeBlock(block); }
  } scratch_guard{impl_->pool.get(), scratch_block};

  const int64_t scratch_slot =
      static_cast<int64_t>(scratch_block) * impl_->pool->block_size();

  ForwardBatch probe;
  probe.num_seqs = num_seqs;
  probe.max_blocks_per_seq = max_blocks_per_seq;
  probe.block_table.assign(
      static_cast<size_t>(num_seqs * max_blocks_per_seq), scratch_block);

  // The probe stands at the *end* of the longest sequence this shape can serve,
  // not at position 0.
  //
  // Everything sized on demand has to reach its final size before capture,
  // because capture records addresses and a later reallocation strands them.
  // The index buffers FlashInfer reads are sized from how many blocks the batch
  // spans, and a probe at position 0 spans exactly one block per sequence -- so
  // capturing there sized them for the smallest batch imaginable, and the first
  // real request with a multi-block prompt grew them and left every captured
  // graph reading freed memory. Standing at the last position instead makes the
  // probe span `max_blocks_per_seq` blocks, which is the most any batch of this
  // shape will ever need.
  const int64_t last_position = max_blocks_per_seq * impl_->pool->block_size() - 1;

  for (int64_t s = 0; s < num_seqs; ++s) {
    probe.token_ids.push_back(0);
    probe.positions.push_back(static_cast<int32_t>(last_position));
    probe.seq_of_token.push_back(static_cast<int32_t>(s));
    probe.slots.push_back(static_cast<int32_t>(scratch_slot));
    // One logits row per sequence, which is what a decode batch asks for. A
    // probe requesting a single row would bake a one-row sampler into a graph
    // meant to serve a batch of N.
    probe.logits_indices.push_back(static_cast<int32_t>(s));
  }

  // The sampler is part of the body and its launch dimensions come from this,
  // so it has to be set before capture or the graph records no sampling at all.
  impl_->sampled_count = num_seqs;

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


// The M2 full-recompute path predates the KV cache and runs everything on the
// default stream. It is a reference and debugging path, not a serving one, and
// it was never routed through the precision selector -- so after quantization
// its projections would read weight buffers that no longer exist. Refusing is
// the honest boundary; making it FP8-capable means moving it onto the model's
// stream, which is a change worth making only if something needs it.
Status Qwen2Model::RequireBf16Weights() const {
  if (!impl_->weights_f8) return OkStatus();

  return FailedPreconditionError(
      "Forward()/ForwardLastLogits() run the bf16 recompute path and the "
      "weights have been quantized to FP8; use Step() with a KV cache");
}

Status Qwen2Model::QuantizeWeightsToF8() {
  if (impl_->weights_f8) return OkStatus();

  if (!impl_->graphs.empty()) {
    // A captured graph holds the bf16 weight addresses and the bf16 kernels.
    // Requantizing under it would replay against freed memory.
    return FailedPreconditionError(
        "cannot quantize weights after capturing a graph; quantize first, then "
        "capture");
  }

  // One float per quantized tensor, allocated together so the pointers stay
  // stable for the life of the model.
  const int64_t layers = static_cast<int64_t>(impl_->layers.size());

  INFERX_ASSIGN_OR_RETURN(
      DeviceBuffer scales,
      DeviceBuffer::Allocate(static_cast<size_t>(layers) * 4 * sizeof(float),
                             DeviceId::Cuda(0)));

  auto* scale_base = reinterpret_cast<float*>(scales.data());

  const auto quantize = [&](const TensorView& src, TensorView* dst,
                            float** scale, int index) -> Status {
    INFERX_ASSIGN_OR_RETURN(
        DeviceBuffer buf,
        DeviceBuffer::Allocate(static_cast<size_t>(src.Numel()),
                               DeviceId::Cuda(0)));

    impl_->f8_buffers.push_back(std::move(buf));

    INFERX_ASSIGN_OR_RETURN(
        *dst, TensorView::Create(impl_->f8_buffers.back().data(),
                                 DataType::kFloat8E4M3FN, src.GetShape(),
                                 DeviceId::Cuda(0)));

    *scale = scale_base + index;

    // The single-block kernel, which is slow on a 90 MB tensor -- one SM
    // cannot saturate memory, so this costs on the order of a second across
    // all 144 tensors. Paid once at load, which is the right place for it; a
    // multi-block variant would be the fix if load time ever mattered.
    return kernels::QuantizeToF8E4M3Dynamic(src, *dst, *scale, impl_->stream);
  };

  int index = 0;
  for (LayerWeights& w : impl_->layers) {
    INFERX_RETURN_IF_ERROR(quantize(w.qkv_w, &w.qkv_w8, &w.qkv_s, index++));
    INFERX_RETURN_IF_ERROR(quantize(w.o_w, &w.o_w8, &w.o_s, index++));
    INFERX_RETURN_IF_ERROR(
        quantize(w.gate_up_w, &w.gate_up_w8, &w.gate_up_s, index++));
    INFERX_RETURN_IF_ERROR(quantize(w.down_w, &w.down_w8, &w.down_s, index++));
  }

  INFERX_CUDA_RETURN_IF_ERROR(cudaStreamSynchronize(impl_->stream));

  impl_->f8_buffers.push_back(std::move(scales));

  INFERX_ASSIGN_OR_RETURN(
      impl_->act_scales,
      DeviceBuffer::Allocate(4 * sizeof(float), DeviceId::Cuda(0)));

  // Release the bf16 originals of exactly the tensors that were quantized, and
  // only after every quantization has landed. Clearing the whole vector would
  // free the embedding, the final norm and every layernorm weight along with
  // them -- all still live, all still bf16, all still pointed at by views that
  // would then dangle.
  for (const LayerWeights& w : impl_->layers) {
    for (const int index : {w.qkv_buf, w.o_buf, w.gate_up_buf, w.down_buf}) {
      if (index >= 0 &&
          index < static_cast<int>(impl_->weight_buffers.size())) {
        impl_->weight_buffers[static_cast<size_t>(index)].Reset();
      }
    }
  }

  impl_->weight_bytes = 0;
  for (const DeviceBuffer& b : impl_->weight_buffers) {
    impl_->weight_bytes += b.size();
  }
  for (const DeviceBuffer& b : impl_->f8_buffers) impl_->weight_bytes += b.size();

  impl_->weights_f8 = true;

  // Now that the flag is set this does real work; if no forward has run yet it
  // is a no-op and the first EnsureCapacity will size it.
  INFERX_RETURN_IF_ERROR(impl_->EnsureActivationF8(impl_->capacity_tokens));

  return OkStatus();
}

bool Qwen2Model::weights_are_f8() const { return impl_->weights_f8; }

Status Qwen2Model::EnableDeviceSampling(int64_t max_rows) {
  // Refused after capture, and this one is worth spelling out: a graph records
  // the kernels present when it was taken. Enabling sampling afterwards leaves
  // every replay running the body that has no sampler in it -- so no token is
  // written, no event is recorded, and AwaitStep returns instantly on an event
  // that was never signalled. Nothing errors; generation simply produces
  // whatever was in the buffer, at an impossible rate. It was found because a
  // benchmark read 7952 tokens/second.
  if (!impl_->graphs.empty()) {
    return FailedPreconditionError(
        "cannot enable device sampling after capturing a graph; enable it "
        "first, then capture");
  }

  if (max_rows <= 0) {
    return InvalidArgumentError("max_rows must be positive, got ", max_rows);
  }

  INFERX_ASSIGN_OR_RETURN(
      impl_->sampled_ids,
      DeviceBuffer::Allocate(static_cast<size_t>(max_rows) * sizeof(int32_t),
                             DeviceId::Cuda(0)));
  INFERX_ASSIGN_OR_RETURN(
      impl_->sample_slots,
      DeviceBuffer::Allocate(static_cast<size_t>(max_rows) * sizeof(int32_t),
                             DeviceId::Cuda(0)));
  INFERX_ASSIGN_OR_RETURN(
      impl_->sample_rows,
      DeviceBuffer::Allocate(static_cast<size_t>(max_rows) * sizeof(int32_t),
                             DeviceId::Cuda(0)));

  if (impl_->pinned_sampled != nullptr) cudaFreeHost(impl_->pinned_sampled);
  INFERX_CUDA_RETURN_IF_ERROR(
      cudaHostAlloc(&impl_->pinned_sampled,
                    static_cast<size_t>(max_rows) * sizeof(int32_t),
                    cudaHostAllocDefault));

  if (impl_->sampled_ready == nullptr) {
    INFERX_CUDA_RETURN_IF_ERROR(cudaEventCreate(&impl_->sampled_ready));
  }

  impl_->sample_on_device = true;

  return OkStatus();
}

Status Qwen2Model::StepAsync(const ForwardBatch& batch) {
  if (!impl_->sample_on_device) {
    return FailedPreconditionError(
        "StepAsync needs device sampling; call EnableDeviceSampling first");
  }

  if (impl_->pool == nullptr) {
    return FailedPreconditionError("StepAsync requires a KV cache");
  }

  const int64_t total_slots =
      impl_->pool->num_blocks() * impl_->pool->block_size();
  INFERX_RETURN_IF_ERROR(batch.Validate(impl_->config.vocab_size, total_slots));

  impl_->sampled_count = static_cast<int64_t>(batch.logits_indices.size());

  INFERX_RETURN_IF_ERROR(impl_->PrepareBatchInputs(batch));

  // Where each sampled token belongs in the next step's token buffer. For
  // decode that is the same index the sequence occupies now, since one token
  // per sequence goes in and one comes out.
  {
    // Which logits rows to sample, and where each result belongs in the next
    // step's token buffer. For decode these coincide -- one token per sequence
    // in, one out, same index -- but a prefill samples its last row and writes
    // to its sequence's slot, so they are uploaded separately.
    const std::vector<int32_t> rows(batch.logits_indices.begin(),
                                    batch.logits_indices.end());

    std::vector<int32_t> slots(rows.size());
    for (size_t i = 0; i < rows.size(); ++i) {
      slots[i] = batch.seq_of_token[static_cast<size_t>(rows[i])];
    }

    INFERX_CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(
        impl_->sample_rows.data(), rows.data(), rows.size() * sizeof(int32_t),
        cudaMemcpyHostToDevice, impl_->stream));
    INFERX_CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(
        impl_->sample_slots.data(), slots.data(),
        slots.size() * sizeof(int32_t), cudaMemcpyHostToDevice,
        impl_->stream));
  }

  const int64_t tokens = batch.num_tokens();

  cudaGraphExec_t graph =
      impl_->FindGraph(tokens, batch.num_seqs, batch.max_blocks_per_seq);

  if (graph != nullptr) {
    INFERX_CUDA_RETURN_IF_ERROR(cudaGraphLaunch(graph, impl_->stream));
  } else {
    INFERX_RETURN_IF_ERROR(impl_->LaunchDecodeBody(
        tokens, batch.num_seqs, batch.max_blocks_per_seq));
  }

  // Recorded here rather than inside the body, so it is a real host-visible
  // event rather than a node buried in the graph. \see LaunchDecodeBody.
  INFERX_CUDA_RETURN_IF_ERROR(
      cudaEventRecord(impl_->sampled_ready, impl_->stream));

  // No synchronize. That absence is the entire feature.
  return OkStatus();
}

Status Qwen2Model::AwaitStep(std::vector<int32_t>* out_tokens) {
  if (impl_->sampled_count == 0) {
    out_tokens->clear();
    return OkStatus();
  }

  INFERX_CUDA_RETURN_IF_ERROR(cudaEventSynchronize(impl_->sampled_ready));

  const auto* ids = static_cast<const int32_t*>(impl_->pinned_sampled);
  out_tokens->assign(ids, ids + impl_->sampled_count);

  return OkStatus();
}

double Qwen2Model::last_step_device_ms() const {
  return impl_->last_device_ms;
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
    INFERX_RETURN_IF_ERROR(DownloadLogitsPinned(
        impl_->logits.buf, index * vocab, vocab, impl_->pinned_logits,
        impl_->stream, &row));
    out_logits->insert(out_logits->end(), row.begin(), row.end());
  }

  return OkStatus();
}

Status Qwen2Model::ForwardLastLogits(const std::vector<int32_t>& token_ids,
                                     std::vector<float>* out_logits) {
  INFERX_RETURN_IF_ERROR(RequireBf16Weights());
  INFERX_RETURN_IF_ERROR(ValidateIds(token_ids, impl_->config.vocab_size));

  int64_t tokens = 0;
  INFERX_RETURN_IF_ERROR(impl_->RunForward(token_ids, &tokens));

  const int64_t vocab = impl_->config.vocab_size;

  return DownloadLogitsPinned(impl_->logits.buf, (tokens - 1) * vocab, vocab,
                              impl_->pinned_logits, impl_->stream, out_logits);
}

}  // namespace inferx::model
