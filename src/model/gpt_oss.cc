#include "inferx/model/gpt_oss.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <utility>

#include <cuda_runtime.h>

#include "absl/strings/str_cat.h"
#include "inferx/core/cuda_utils.h"
#include "inferx/core/device_buffer.h"
#include "inferx/kernels/gemm.h"
#include "inferx/kernels/gpt_oss.h"
#include "inferx/kernels/layers.h"
#include "inferx/kernels/mxfp4.h"
#include "inferx/model/moe_ffn.h"

namespace inferx::model {
namespace {

/// Wall-clock phase accumulator for `Forward`, gated by the
/// `INFERX_GPTOSS_PROFILE` env var. Measures host time across the layer loop,
/// which -- because every `cudaMemcpy` in the MoE path is the blocking form --
/// is also where the host sits waiting on PCIe. The breakdown names the phase
/// to fix rather than the kernel to tune: if `dequant_upload` dominates, the
/// fix is async double-buffering; if `moe_forward` dominates, it is the GEMM
/// (X3); if `sync` dominates, the host is already launching faster than the
/// device can consume and the win is elsewhere.
struct ForwardProfile {
  using Clock = std::chrono::steady_clock;
  Clock::time_point last = Clock::now();
  double dequant = 0;      // DequantizeLayerExperts (now: dequantize kernel only)
  double moe_forward = 0;  // MoeFfn::Forward
  double attn = 0;         // the full attention block
  int64_t layers = 0;

  void tick(double& bucket) {
    const auto now = Clock::now();
    bucket += std::chrono::duration<double, std::milli>(now - last).count();
    last = now;
  }

  void report(int64_t tokens) const {
    const double total = dequant + moe_forward + attn;
    std::fprintf(stderr,
        "[gpt-oss profile] %lld tokens, %lld layers, total %.0f ms\n"
        "  dequant (device-resident MXFP4 -> bf16): %8.1f ms  (%4.1f%%)\n"
        "  moe_forward:                             %8.1f ms  (%4.1f%%)\n"
        "  attention (rmsn+qkv+rope+att             %8.1f ms  (%4.1f%%)\n"
        "  +sink+oproj+residual):\n",
        static_cast<long long>(tokens), static_cast<long long>(layers), total,
        dequant, 100 * dequant / total, moe_forward,
        100 * moe_forward / total, attn, 100 * attn / total);
  }
};

constexpr DataType kBf16 = DataType::kBFloat16;
constexpr int kMxfp4BytesPerBlock = 16;
constexpr int kMxfp4ValuesPerBlock = 32;

struct Scratch {
  DeviceBuffer buf;

  StatusOr<TensorView> View(DataType dtype, const Shape& shape) const {
    return TensorView::Create(buf.data(), dtype, shape, DeviceId::Cuda(0));
  }
};

Status Grow(Scratch* s, size_t bytes) {
  if (s->buf.size() >= bytes) return OkStatus();
  INFERX_ASSIGN_OR_RETURN(s->buf,
                          DeviceBuffer::Allocate(bytes, DeviceId::Cuda(0)));
  return OkStatus();
}

// Uploads a host tensor to the device and returns a view over the copy. The
// buffer is appended to `keep`, which owns it for the model's lifetime.
StatusOr<TensorView> Upload(std::vector<DeviceBuffer>* keep,
                            const Tensor& host) {
  const size_t bytes = DataTypeByteSize(host.dtype(), host.numel());

  INFERX_ASSIGN_OR_RETURN(DeviceBuffer buf,
                          DeviceBuffer::Allocate(bytes, DeviceId::Cuda(0)));
  INFERX_CUDA_RETURN_IF_ERROR(
      cudaMemcpy(buf.data(), host.data(), bytes, cudaMemcpyHostToDevice));

  keep->push_back(std::move(buf));

  return TensorView::Create(keep->back().data(), host.dtype(),
                            host.shape(), DeviceId::Cuda(0));
}

// Uploads a `[experts, 2·inter]` bias with its last axis de-interleaved, so it
// lines up with weights that `DequantizeMxfp4GateUpToBf16` already split into
// `[gate | up]`. Done on the host because it is 368 KB per layer and happens
// once per layer per call, against 423 MB of weights moving beside it.
StatusOr<TensorView> UploadDeinterleaved(std::vector<DeviceBuffer>* keep,
                                         const Tensor& host) {
  if (host.rank() != 2 || host.dim(1) % 2 != 0) {
    return InvalidArgumentError("gate_up bias must be [experts, even], got ",
                                host.shape().ToString());
  }

  const int64_t experts = host.dim(0);
  const int64_t width = host.dim(1);
  const int64_t half = width / 2;

  const auto* src = static_cast<const uint16_t*>(host.data());
  std::vector<uint16_t> permuted(static_cast<size_t>(experts * width));

  for (int64_t e = 0; e < experts; ++e) {
    for (int64_t j = 0; j < half; ++j) {
      permuted[static_cast<size_t>(e * width + j)] =
          src[static_cast<size_t>(e * width + 2 * j)];
      permuted[static_cast<size_t>(e * width + half + j)] =
          src[static_cast<size_t>(e * width + 2 * j + 1)];
    }
  }

  const size_t bytes = permuted.size() * sizeof(uint16_t);

  INFERX_ASSIGN_OR_RETURN(DeviceBuffer buf,
                          DeviceBuffer::Allocate(bytes, DeviceId::Cuda(0)));
  INFERX_CUDA_RETURN_IF_ERROR(
      cudaMemcpy(buf.data(), permuted.data(), bytes, cudaMemcpyHostToDevice));

  keep->push_back(std::move(buf));

  return TensorView::Create(keep->back().data(), host.dtype(), host.shape(),
                            DeviceId::Cuda(0));
}

// One layer's non-expert weights, resident on the device.
struct LayerWeights {
  TensorView input_norm;
  TensorView post_norm;

  TensorView q_w, q_b;
  TensorView k_w, k_b;
  TensorView v_w, v_b;
  TensorView o_w, o_b;
  TensorView sinks;

  TensorView router_w, router_b;

  // The packed MXFP4 expert weights and the bf16 biases are uploaded to the
  // device ONCE at Load and stay resident for the model's lifetime. The
  // profiling harness (INFERX_GPTOSS_PROFILE) showed the per-call blocking
  // host->device upload of these was 98.6% of Forward -- ~15 s of a 15 s
  // forward -- because the experts re-crossed PCIe every layer of every call.
  // Resident, that upload is paid once at load; per-call work drops to just
  // the dequantize kernel reading device-resident MXFP4 into the reusable
  // bf16 scratch.
  //
  // The host Tensors below are kept only because Checkpoint already maps them
  // and the device upload reads from them once; after Load they are not
  // touched again.
  Tensor gate_up_blocks, gate_up_scales, gate_up_bias;
  Tensor down_blocks, down_scales, down_bias;

  // Device-resident views over the same data, set at Load. DequantizeLayerExperts
  // reads these instead of uploading, and Forward reads the biases from here
  // instead of re-uploading per layer.
  TensorView gate_up_blocks_dev, gate_up_scales_dev;
  TensorView down_blocks_dev, down_scales_dev;
  TensorView gate_up_bias_dev, down_bias_dev;
};

}  // namespace

struct GptOssModel::Impl {
  Impl(ModelConfig c, Checkpoint ck, kernels::CublasLtGemm g)
      : config(std::move(c)), ckpt(std::move(ck)), gemm(std::move(g)) {}

  ModelConfig config;
  Checkpoint ckpt;

  std::vector<DeviceBuffer> weight_buffers;
  std::vector<LayerWeights> layers;

  TensorView embed;
  TensorView final_norm;
  TensorView lm_head;

  kernels::CublasLtGemm gemm;
  std::unique_ptr<MoeFfn> moe;

  // YaRN's frequency table, computed once on the host.
  Scratch inv_freq;
  float attn_factor = 1.0f;

  // Per-call activations, grown on demand.
  int64_t capacity_tokens = 0;
  Scratch ids, positions;
  Scratch hidden, residual, normed;
  Scratch qkv_q, qkv_k, qkv_v, attn_out, lse;
  Scratch proj, moe_out, logits;

  // The MXFP4 staging scratch is gone: experts are device-resident now and
  // DequantizeLayerExperts reads them in place. What remains is the dequantized
  // bf16 scratch, one layer's worth, reused.
  Scratch expert_gate_up, expert_down;

  // --- Paged KV (Phase 4a: continuous batching) ---------------------------
  //
  // Forward() is the batch-1 full-recompute reference and uses none of this.
  // Step() writes each layer's K/V here and reads it back, which is what lets
  // the scheduler batch multiple sequences into one Step. The 12 sliding
  // layers are served as full-attention for now (every token cached, none
  // evicted) -- a memory cost, not a correctness one; R-C's per-layer
  // lifetime is the follow-up.
  std::unique_ptr<KvBlockPool> pool;

  // Per-step index buffers, device-resident. Grown with the token / sequence
  // count, rewritten every step. Uploaded with plain cudaMemcpy rather than
  // pinned staging because the MoE upload dominates the step and the index
  // traffic is noise beside it.
  DeviceBuffer slots_buf, seq_of_token_buf, block_table_buf;
  int64_t block_table_capacity = 0;

  // Longest key sequence any query in this batch attends over, computed in
  // PrepareBatchInputs. Sizes the reference attention kernel's shared-memory
  // tile for the full-attention layers; the sliding layers cap it at
  // `sliding_window` instead.
  int64_t batch_max_context = 0;

  Status EnsureCapacity(int64_t tokens);
  Status DequantizeLayerExperts(const LayerWeights& layer,
                                TensorView* gate_up, TensorView* down);

  // Step's two halves, split for the same reason Qwen2 splits them: the first
  // is host-side index staging (never inside a captured region, when graphs
  // arrive) and the second is device work.
  Status PrepareBatchInputs(const ForwardBatch& batch);
  Status RunPagedForward(const ForwardBatch& batch);
};

Status GptOssModel::Impl::EnsureCapacity(int64_t tokens) {
  if (tokens <= capacity_tokens) return OkStatus();

  const size_t sz = DataTypeByteSize(kBf16, 1);
  const int64_t h = config.hidden_size;
  const int64_t heads = config.num_attention_heads;
  const int64_t hd = config.head_dim;

  INFERX_RETURN_IF_ERROR(Grow(&ids, sizeof(int32_t) * tokens));
  INFERX_RETURN_IF_ERROR(Grow(&positions, sizeof(int32_t) * tokens));
  INFERX_RETURN_IF_ERROR(Grow(&hidden, sz * tokens * h));
  INFERX_RETURN_IF_ERROR(Grow(&residual, sz * tokens * h));
  INFERX_RETURN_IF_ERROR(Grow(&normed, sz * tokens * h));
  INFERX_RETURN_IF_ERROR(Grow(&qkv_q, sz * tokens * config.q_dim()));
  INFERX_RETURN_IF_ERROR(Grow(&qkv_k, sz * tokens * config.kv_dim()));
  INFERX_RETURN_IF_ERROR(Grow(&qkv_v, sz * tokens * config.kv_dim()));
  INFERX_RETURN_IF_ERROR(Grow(&attn_out, sz * tokens * heads * hd));
  INFERX_RETURN_IF_ERROR(Grow(&lse, sizeof(float) * tokens * heads));
  INFERX_RETURN_IF_ERROR(Grow(&proj, sz * tokens * h));
  INFERX_RETURN_IF_ERROR(Grow(&moe_out, sz * tokens * h));
  INFERX_RETURN_IF_ERROR(Grow(&logits, sz * tokens * config.vocab_size));

  capacity_tokens = tokens;
  return OkStatus();
}

// Uploads one layer's packed experts and decodes them into the bf16 scratch.
//
// This is the deliberately slow step, and the whole reason the class can run a
// 21B model on a 16 GB card without a 4-bit GEMM. `gate_up` is de-interleaved
// on the way through, so what comes out is `[gate | up]` like every other model
// here rather than gpt-oss's alternating layout.
Status GptOssModel::Impl::DequantizeLayerExperts(const LayerWeights& layer,
                                                 TensorView* gate_up,
                                                 TensorView* down) {
  const int64_t experts = config.num_experts;
  const int64_t inter = config.moe_intermediate_size;
  const int64_t h = config.hidden_size;

  // Reads the resident device-resident MXFP4 (uploaded once at Load) and
  // dequantizes into the reusable bf16 scratch. No host->device copy -- that
  // copy was 98.6% of Forward under the profiler, and it is gone now.
  auto decode = [&](const TensorView& blocks_dev,
                    const TensorView& scales_dev, int64_t rows,
                    bool deinterleave, Scratch* dest_scratch,
                    TensorView* out) -> Status {
    // The resident views carry the checkpoint's rank-4/3 shape
    // ([E, rows, num_blocks, 16] / [E, rows, num_blocks]); the kernel wants the
    // expert axis flattened in. Row-major storage makes the reshape a view, not
    // a copy -- the bytes are identical, only the indexing changes.
    INFERX_ASSIGN_OR_RETURN(
        const TensorView blocks_v,
        blocks_dev.Reshape(Shape({experts * rows,
                                  blocks_dev.Dim(blocks_dev.Rank() - 2),
                                  kMxfp4BytesPerBlock})));
    INFERX_ASSIGN_OR_RETURN(
        const TensorView scales_v,
        scales_dev.Reshape(Shape(
            {experts * rows, scales_dev.Dim(scales_dev.Rank() - 1)})));

    const int64_t num_blocks = blocks_dev.Dim(blocks_dev.Rank() - 2);
    const int64_t width = num_blocks * kMxfp4ValuesPerBlock;

    INFERX_RETURN_IF_ERROR(
        Grow(dest_scratch, DataTypeByteSize(kBf16, experts * rows * width)));

    // The de-interleave splits even/odd rows into the two halves of *its*
    // output, so it has to see one expert at a time -- flattening across
    // experts would interleave expert 0's `up` rows with expert 1's `gate`.
    if (deinterleave) {
      for (int64_t e = 0; e < experts; ++e) {
        INFERX_ASSIGN_OR_RETURN(const TensorView eb,
                                blocks_v.Slice(e * rows, (e + 1) * rows));
        INFERX_ASSIGN_OR_RETURN(const TensorView es,
                                scales_v.Slice(e * rows, (e + 1) * rows));

        INFERX_ASSIGN_OR_RETURN(
            const TensorView all,
            dest_scratch->View(kBf16, Shape({experts * rows, width})));
        INFERX_ASSIGN_OR_RETURN(const TensorView eo,
                                all.Slice(e * rows, (e + 1) * rows));

        INFERX_RETURN_IF_ERROR(
            kernels::DequantizeMxfp4GateUpToBf16(eb, es, eo));
      }
    } else {
      INFERX_ASSIGN_OR_RETURN(
          const TensorView all,
          dest_scratch->View(kBf16, Shape({experts * rows, width})));
      INFERX_RETURN_IF_ERROR(
          kernels::DequantizeMxfp4ToBf16(blocks_v, scales_v, all));
    }

    INFERX_ASSIGN_OR_RETURN(
        *out, dest_scratch->View(kBf16, Shape({experts, rows, width})));

    return OkStatus();
  };

  INFERX_RETURN_IF_ERROR(decode(layer.gate_up_blocks_dev,
                                layer.gate_up_scales_dev, 2 * inter,
                                /*deinterleave=*/true, &expert_gate_up,
                                gate_up));

  INFERX_RETURN_IF_ERROR(decode(layer.down_blocks_dev, layer.down_scales_dev,
                                h, /*deinterleave=*/false, &expert_down, down));

  return OkStatus();
}

// Host-side half of Step: sizes the index buffers and uploads this batch's
// indices. No device work of its own -- the Qwen2 path splits the same way so
// that the device half can be recorded into a graph later, and even without
// graphs the split keeps the index staging out of the launch-critical path.
Status GptOssModel::Impl::PrepareBatchInputs(const ForwardBatch& batch) {
  const int64_t tokens = batch.num_tokens();

  INFERX_RETURN_IF_ERROR(EnsureCapacity(tokens));

  // The block table scales with sequences rather than tokens, so it has its own
  // capacity. The per-token arrays share a single grow check.
  const int64_t table_elems = batch.num_seqs * batch.max_blocks_per_seq;
  if (table_elems > block_table_capacity) {
    INFERX_ASSIGN_OR_RETURN(
        block_table_buf,
        DeviceBuffer::Allocate(static_cast<size_t>(table_elems) * sizeof(int32_t),
                               DeviceId::Cuda(0)));
    block_table_capacity = table_elems;
  }

  const size_t tok_bytes = static_cast<size_t>(tokens) * sizeof(int32_t);

  if (slots_buf.size() < tok_bytes) {
    INFERX_ASSIGN_OR_RETURN(slots_buf,
                            DeviceBuffer::Allocate(tok_bytes, DeviceId::Cuda(0)));
    INFERX_ASSIGN_OR_RETURN(seq_of_token_buf,
                            DeviceBuffer::Allocate(tok_bytes, DeviceId::Cuda(0)));
  }

  // Plain cudaMemcpy rather than the Qwen2 path's pinned staging. A gpt-oss
  // step moves ~10 GB of expert weights over PCIe per layer, so the few KB of
  // index traffic -- and the few hundred microseconds of host time a pageable
  // copy costs -- is noise. Keeping it simple here is worth not reproducing the
  // pinned-buffer plumbing until the MoE upload itself is gone (Phase 4).
  if (!batch.tokens_from_device) {
    INFERX_CUDA_RETURN_IF_ERROR(cudaMemcpy(
        ids.buf.data(), batch.token_ids.data(), tok_bytes, cudaMemcpyHostToDevice));
  }
  INFERX_CUDA_RETURN_IF_ERROR(cudaMemcpy(
      positions.buf.data(), batch.positions.data(), tok_bytes,
      cudaMemcpyHostToDevice));
  INFERX_CUDA_RETURN_IF_ERROR(cudaMemcpy(
      slots_buf.data(), batch.slots.data(), tok_bytes, cudaMemcpyHostToDevice));
  INFERX_CUDA_RETURN_IF_ERROR(cudaMemcpy(
      seq_of_token_buf.data(), batch.seq_of_token.data(), tok_bytes,
      cudaMemcpyHostToDevice));
  INFERX_CUDA_RETURN_IF_ERROR(cudaMemcpy(
      block_table_buf.data(), batch.block_table.data(),
      static_cast<size_t>(table_elems) * sizeof(int32_t),
      cudaMemcpyHostToDevice));

  // Longest key sequence any query attends over. For full-attention layers this
  // sizes the reference kernel's shared-memory tile; for sliding layers it is
  // capped at `sliding_window` per launch instead.
  batch_max_context = 0;
  for (const int32_t p : batch.positions) {
    batch_max_context = std::max<int64_t>(batch_max_context, p + 1);
  }

  return OkStatus();
}

// Device half of Step: the whole paged forward, structured to match Forward()
// layer-for-layer so the two can be diffed when something disagrees. The three
// differences from Forward are exactly the paging substitutions:
//   * positions come from the batch (per-token, per-sequence), not 0..n-1;
//   * K/V are written into the paged cache after RoPE, not kept in scratch;
//   * attention reads the cache through the block table, not the scratch K/V,
//     and reports its lse so the per-head sink can be folded in afterwards.
Status GptOssModel::Impl::RunPagedForward(const ForwardBatch& batch) {
  Impl& m = *this;
  const ModelConfig& c = m.config;
  const int64_t tokens = batch.num_tokens();
  const int64_t h = c.hidden_size;
  const int64_t heads = c.num_attention_heads;
  const int64_t kv_heads = c.num_key_value_heads;
  const int64_t hd = c.head_dim;

  const auto i32 = [&](const DeviceBuffer& buf,
                       const Shape& shape) -> StatusOr<TensorView> {
    return TensorView::Create(buf.data(), DataType::kInt32, shape,
                              DeviceId::Cuda(0));
  };

  INFERX_ASSIGN_OR_RETURN(const TensorView ids_v,
                          i32(m.ids.buf, Shape({tokens})));
  INFERX_ASSIGN_OR_RETURN(const TensorView pos_v,
                          i32(m.positions.buf, Shape({tokens})));
  INFERX_ASSIGN_OR_RETURN(const TensorView slots_v,
                          i32(m.slots_buf, Shape({tokens})));
  INFERX_ASSIGN_OR_RETURN(const TensorView seq_v,
                          i32(m.seq_of_token_buf, Shape({tokens})));
  INFERX_ASSIGN_OR_RETURN(
      const TensorView table_v,
      i32(m.block_table_buf,
          Shape({batch.num_seqs, batch.max_blocks_per_seq})));

  INFERX_ASSIGN_OR_RETURN(const TensorView x,
                          m.hidden.View(kBf16, Shape({tokens, h})));
  INFERX_ASSIGN_OR_RETURN(const TensorView resid,
                          m.residual.View(kBf16, Shape({tokens, h})));
  INFERX_ASSIGN_OR_RETURN(const TensorView normed,
                          m.normed.View(kBf16, Shape({tokens, h})));
  INFERX_ASSIGN_OR_RETURN(
      const TensorView q, m.qkv_q.View(kBf16, Shape({tokens, heads, hd})));
  INFERX_ASSIGN_OR_RETURN(
      const TensorView k, m.qkv_k.View(kBf16, Shape({tokens, kv_heads, hd})));
  INFERX_ASSIGN_OR_RETURN(
      const TensorView v, m.qkv_v.View(kBf16, Shape({tokens, kv_heads, hd})));
  INFERX_ASSIGN_OR_RETURN(
      const TensorView attn, m.attn_out.View(kBf16, Shape({tokens, heads, hd})));
  INFERX_ASSIGN_OR_RETURN(
      const TensorView lse_v,
      m.lse.View(DataType::kFloat, Shape({tokens, heads})));
  INFERX_ASSIGN_OR_RETURN(const TensorView proj,
                          m.proj.View(kBf16, Shape({tokens, h})));
  INFERX_ASSIGN_OR_RETURN(const TensorView moe_out,
                          m.moe_out.View(kBf16, Shape({tokens, h})));
  INFERX_ASSIGN_OR_RETURN(
      const TensorView inv_freq,
      m.inv_freq.View(DataType::kFloat, Shape({hd / 2})));

  // Flat views for the projections (2-D) and the cache append (3-D per head).
  INFERX_ASSIGN_OR_RETURN(const TensorView q2,
                          q.Reshape(Shape({tokens, c.q_dim()})));
  INFERX_ASSIGN_OR_RETURN(const TensorView k2,
                          k.Reshape(Shape({tokens, c.kv_dim()})));
  INFERX_ASSIGN_OR_RETURN(const TensorView v2,
                          v.Reshape(Shape({tokens, c.kv_dim()})));
  INFERX_ASSIGN_OR_RETURN(const TensorView attn2,
                          attn.Reshape(Shape({tokens, heads * hd})));

  INFERX_RETURN_IF_ERROR(kernels::EmbeddingLookup(m.embed, ids_v, x));

  const float scale = 1.0f / std::sqrt(static_cast<float>(hd));

  for (int64_t layer = 0; layer < c.num_hidden_layers; ++layer) {
    const LayerWeights& w = m.layers[static_cast<size_t>(layer)];

    // --- attention ---------------------------------------------------------
    INFERX_CUDA_RETURN_IF_ERROR(cudaMemcpy(
        resid.Data(), x.Data(), DataTypeByteSize(kBf16, tokens * h),
        cudaMemcpyDeviceToDevice));

    INFERX_RETURN_IF_ERROR(kernels::RmsNorm(x, w.input_norm, normed,
                                            static_cast<float>(c.rms_norm_eps)));

    INFERX_RETURN_IF_ERROR(m.gemm.LinearBF16(normed, w.q_w, q2));
    INFERX_RETURN_IF_ERROR(kernels::AddBiasInPlace(q2, w.q_b));
    INFERX_RETURN_IF_ERROR(m.gemm.LinearBF16(normed, w.k_w, k2));
    INFERX_RETURN_IF_ERROR(kernels::AddBiasInPlace(k2, w.k_b));
    INFERX_RETURN_IF_ERROR(m.gemm.LinearBF16(normed, w.v_w, v2));
    INFERX_RETURN_IF_ERROR(kernels::AddBiasInPlace(v2, w.v_b));

    INFERX_RETURN_IF_ERROR(kernels::RotaryEmbeddingFromTable(
        q, k, pos_v, inv_freq, m.attn_factor));

    // Cache write *after* RoPE: a cached key is stored at the position it was
    // rotated for, so it never needs re-rotating when read back.
    INFERX_ASSIGN_OR_RETURN(const TensorView k_cache, m.pool->KeyCache(layer));
    INFERX_ASSIGN_OR_RETURN(const TensorView v_cache, m.pool->ValueCache(layer));

    INFERX_RETURN_IF_ERROR(
        kernels::AppendToKvCache(k, v, k_cache, v_cache, slots_v));

    // Per-layer window: gpt-oss alternates full and sliding (128). The tile is
    // sized from `window` on a sliding layer, which is also the smem win.
    const int64_t window =
        c.IsSlidingLayer(layer) ? c.sliding_window : 0;
    const int64_t max_ctx =
        window > 0 ? window : m.batch_max_context;

    INFERX_RETURN_IF_ERROR(kernels::PagedAttentionWithLse(
        q, k_cache, v_cache, table_v, seq_v, pos_v, attn, lse_v, scale, window,
        max_ctx));

    // The sink, folded in using the lse the kernel just wrote. Natural-log
    // convention here (the kernel computes max + ln(sum)), so lse_is_log2 is
    // false -- unlike the FlashInfer path, which is base-2.
    INFERX_RETURN_IF_ERROR(kernels::ApplyAttentionSinks(
        attn, lse_v, w.sinks, /*lse_is_log2=*/false));

    INFERX_RETURN_IF_ERROR(m.gemm.LinearBF16(attn2, w.o_w, proj));
    INFERX_RETURN_IF_ERROR(kernels::AddBiasInPlace(proj, w.o_b));
    INFERX_RETURN_IF_ERROR(kernels::AddInPlace(proj, resid));
    INFERX_CUDA_RETURN_IF_ERROR(cudaMemcpy(
        x.Data(), proj.Data(), DataTypeByteSize(kBf16, tokens * h),
        cudaMemcpyDeviceToDevice));

    // --- MoE ---------------------------------------------------------------
    INFERX_CUDA_RETURN_IF_ERROR(cudaMemcpy(
        resid.Data(), x.Data(), DataTypeByteSize(kBf16, tokens * h),
        cudaMemcpyDeviceToDevice));

    INFERX_RETURN_IF_ERROR(kernels::RmsNorm(
        x, w.post_norm, normed, static_cast<float>(c.rms_norm_eps)));

    MoeWeights mw;
    mw.router = w.router_w;
    mw.router_bias = w.router_b;

    INFERX_RETURN_IF_ERROR(
        m.DequantizeLayerExperts(w, &mw.gate_up, &mw.down));

    // Resident biases -- same change as Forward(). No per-call upload, no
    // bias_keep free, no per-layer sync.
    mw.gate_up_bias = w.gate_up_bias_dev;
    mw.down_bias = w.down_bias_dev;

    INFERX_RETURN_IF_ERROR(m.moe->Forward(normed, mw, moe_out, &m.gemm));

    INFERX_RETURN_IF_ERROR(kernels::AddInPlace(moe_out, resid));
    INFERX_CUDA_RETURN_IF_ERROR(cudaMemcpy(
        x.Data(), moe_out.Data(), DataTypeByteSize(kBf16, tokens * h),
        cudaMemcpyDeviceToDevice));
  }

  INFERX_RETURN_IF_ERROR(kernels::RmsNorm(x, m.final_norm, normed,
                                          static_cast<float>(c.rms_norm_eps)));

  INFERX_ASSIGN_OR_RETURN(
      const TensorView logits,
      m.logits.View(kBf16, Shape({tokens, c.vocab_size})));
  INFERX_RETURN_IF_ERROR(m.gemm.LinearBF16(normed, m.lm_head, logits));

  INFERX_CUDA_RETURN_IF_ERROR(cudaDeviceSynchronize());

  return OkStatus();
}

StatusOr<GptOssModel> GptOssModel::Load(std::string_view dir) {
  INFERX_ASSIGN_OR_RETURN(ModelConfig config, ModelConfig::FromDirectory(dir));

  if (config.architecture != Architecture::kGptOss) {
    return InvalidArgumentError("GptOssModel: checkpoint declares ",
                                ArchitectureName(config.architecture));
  }

  INFERX_ASSIGN_OR_RETURN(Checkpoint ckpt, Checkpoint::Open(dir));
  INFERX_ASSIGN_OR_RETURN(kernels::CublasLtGemm gemm,
                          kernels::CublasLtGemm::Create());

  auto impl = std::make_unique<Impl>(config, std::move(ckpt), std::move(gemm));

  const int64_t h = config.hidden_size;
  const int64_t heads = config.num_attention_heads;
  const int64_t hd = config.head_dim;
  const int64_t experts = config.num_experts;
  const int64_t inter = config.moe_intermediate_size;

  auto get = [&](const std::string& name) { return impl->ckpt.Get(name); };

  INFERX_ASSIGN_OR_RETURN(const Tensor embed, get("model.embed_tokens.weight"));
  INFERX_ASSIGN_OR_RETURN(impl->embed, Upload(&impl->weight_buffers, embed));

  INFERX_ASSIGN_OR_RETURN(const Tensor norm, get("model.norm.weight"));
  INFERX_ASSIGN_OR_RETURN(impl->final_norm,
                          Upload(&impl->weight_buffers, norm));

  // Untied: gpt-oss carries a separate lm_head, which is 1.16 GB of the
  // budget on its own.
  INFERX_ASSIGN_OR_RETURN(const Tensor head, get("lm_head.weight"));
  INFERX_ASSIGN_OR_RETURN(impl->lm_head, Upload(&impl->weight_buffers, head));

  impl->layers.resize(static_cast<size_t>(config.num_hidden_layers));

  for (int64_t i = 0; i < config.num_hidden_layers; ++i) {
    LayerWeights& w = impl->layers[static_cast<size_t>(i)];
    const std::string p = absl::StrCat("model.layers.", i, ".");

    auto up = [&](const std::string& name, TensorView* out) -> Status {
      INFERX_ASSIGN_OR_RETURN(const Tensor t, get(p + name));
      INFERX_ASSIGN_OR_RETURN(*out, Upload(&impl->weight_buffers, t));
      return OkStatus();
    };

    INFERX_RETURN_IF_ERROR(up("input_layernorm.weight", &w.input_norm));
    INFERX_RETURN_IF_ERROR(up("post_attention_layernorm.weight", &w.post_norm));

    INFERX_RETURN_IF_ERROR(up("self_attn.q_proj.weight", &w.q_w));
    INFERX_RETURN_IF_ERROR(up("self_attn.q_proj.bias", &w.q_b));
    INFERX_RETURN_IF_ERROR(up("self_attn.k_proj.weight", &w.k_w));
    INFERX_RETURN_IF_ERROR(up("self_attn.k_proj.bias", &w.k_b));
    INFERX_RETURN_IF_ERROR(up("self_attn.v_proj.weight", &w.v_w));
    INFERX_RETURN_IF_ERROR(up("self_attn.v_proj.bias", &w.v_b));
    INFERX_RETURN_IF_ERROR(up("self_attn.o_proj.weight", &w.o_w));
    INFERX_RETURN_IF_ERROR(up("self_attn.o_proj.bias", &w.o_b));
    INFERX_RETURN_IF_ERROR(up("self_attn.sinks", &w.sinks));

    INFERX_RETURN_IF_ERROR(up("mlp.router.weight", &w.router_w));
    INFERX_RETURN_IF_ERROR(up("mlp.router.bias", &w.router_b));

    // The expert weights are read from the checkpoint once and uploaded to the
    // device here, permanently. The host `get` borrows the checkpoint's mapping
    // (so the read costs no memory until the page is touched); the `Upload`
    // then copies it into `weight_buffers`, which owns it for the model's life.
    INFERX_ASSIGN_OR_RETURN(w.gate_up_blocks,
                            get(p + "mlp.experts.gate_up_proj_blocks"));
    INFERX_ASSIGN_OR_RETURN(w.gate_up_scales,
                            get(p + "mlp.experts.gate_up_proj_scales"));
    INFERX_ASSIGN_OR_RETURN(w.gate_up_bias,
                            get(p + "mlp.experts.gate_up_proj_bias"));
    INFERX_ASSIGN_OR_RETURN(w.down_blocks,
                            get(p + "mlp.experts.down_proj_blocks"));
    INFERX_ASSIGN_OR_RETURN(w.down_scales,
                            get(p + "mlp.experts.down_proj_scales"));
    INFERX_ASSIGN_OR_RETURN(w.down_bias,
                            get(p + "mlp.experts.down_proj_bias"));

    // Device-resident copies. The ~10 GB this uploads across all 24 layers is
    // paid ONCE at load rather than per-layer-per-call, which the profile
    // showed was 98.6% of Forward. gate_up_bias is de-interleaved on the way
    // up so the device layout matches `DequantizeMxfp4GateUpToBf16`'s split
    // output; the rest copy verbatim.
    INFERX_ASSIGN_OR_RETURN(
        w.gate_up_blocks_dev, Upload(&impl->weight_buffers, w.gate_up_blocks));
    INFERX_ASSIGN_OR_RETURN(
        w.gate_up_scales_dev, Upload(&impl->weight_buffers, w.gate_up_scales));
    INFERX_ASSIGN_OR_RETURN(
        w.gate_up_bias_dev,
        UploadDeinterleaved(&impl->weight_buffers, w.gate_up_bias));
    INFERX_ASSIGN_OR_RETURN(
        w.down_blocks_dev, Upload(&impl->weight_buffers, w.down_blocks));
    INFERX_ASSIGN_OR_RETURN(
        w.down_scales_dev, Upload(&impl->weight_buffers, w.down_scales));
    INFERX_ASSIGN_OR_RETURN(w.down_bias_dev,
                            Upload(&impl->weight_buffers, w.down_bias));
  }

  // Experts are now resident on the device; the comment that used to live here
  // about "uploaded lazily below, at first use" is retired, as is the whole
  // per-call upload path.

  // YaRN's frequencies, computed once.
  std::vector<float> inv_freq(static_cast<size_t>(hd / 2));

  if (config.is_yarn()) {
    impl->attn_factor = kernels::ComputeYarnInvFreq(
        hd, config.rope_theta, config.yarn_factor, config.yarn_beta_fast,
        config.yarn_beta_slow, config.yarn_original_max_position,
        config.yarn_truncate, inv_freq.data());
  } else {
    for (int64_t j = 0; j < hd / 2; ++j) {
      inv_freq[static_cast<size_t>(j)] = static_cast<float>(
          std::pow(config.rope_theta,
                   -2.0 * static_cast<double>(j) / static_cast<double>(hd)));
    }
  }

  INFERX_RETURN_IF_ERROR(
      Grow(&impl->inv_freq, sizeof(float) * inv_freq.size()));
  INFERX_CUDA_RETURN_IF_ERROR(cudaMemcpy(impl->inv_freq.buf.data(),
                                         inv_freq.data(),
                                         sizeof(float) * inv_freq.size(),
                                         cudaMemcpyHostToDevice));

  MoeFfn::Config moe_config;
  moe_config.hidden = h;
  moe_config.num_experts = experts;
  moe_config.top_k = config.num_experts_per_tok;
  moe_config.moe_intermediate = inter;
  moe_config.norm_topk_prob = true;  // equals gpt-oss's top-k-then-softmax
  moe_config.activation = MoeFfn::Activation::kGptOssClamped;
  moe_config.swiglu_limit = static_cast<float>(config.swiglu_limit);
  moe_config.swiglu_alpha = static_cast<float>(config.swiglu_alpha);

  INFERX_ASSIGN_OR_RETURN(MoeFfn moe, MoeFfn::Create(moe_config, 1));
  impl->moe = std::make_unique<MoeFfn>(std::move(moe));

  (void)heads;

  return GptOssModel(std::move(impl));
}

GptOssModel::GptOssModel(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
GptOssModel::~GptOssModel() = default;
GptOssModel::GptOssModel(GptOssModel&&) noexcept = default;
GptOssModel& GptOssModel::operator=(GptOssModel&&) noexcept = default;

const ModelConfig& GptOssModel::config() const { return impl_->config; }

Status GptOssModel::Forward(const std::vector<int32_t>& token_ids,
                            std::vector<float>* out_logits) {
  if (token_ids.empty()) {
    return InvalidArgumentError("GptOssModel: empty prompt");
  }

  Impl& m = *impl_;
  const ModelConfig& c = m.config;
  const int64_t tokens = static_cast<int64_t>(token_ids.size());
  const int64_t h = c.hidden_size;
  const int64_t heads = c.num_attention_heads;
  const int64_t kv_heads = c.num_key_value_heads;
  const int64_t hd = c.head_dim;

  for (const int32_t id : token_ids) {
    if (id < 0 || id >= c.vocab_size) {
      return InvalidArgumentError("token id ", id, " is outside [0, ",
                                  c.vocab_size, ")");
    }
  }

  INFERX_RETURN_IF_ERROR(m.EnsureCapacity(tokens));

  std::vector<int32_t> pos(static_cast<size_t>(tokens));
  for (int64_t t = 0; t < tokens; ++t) pos[static_cast<size_t>(t)] = t;

  INFERX_CUDA_RETURN_IF_ERROR(cudaMemcpy(m.ids.buf.data(), token_ids.data(),
                                         sizeof(int32_t) * tokens,
                                         cudaMemcpyHostToDevice));
  INFERX_CUDA_RETURN_IF_ERROR(cudaMemcpy(m.positions.buf.data(), pos.data(),
                                         sizeof(int32_t) * tokens,
                                         cudaMemcpyHostToDevice));

  INFERX_ASSIGN_OR_RETURN(const TensorView ids_v,
                          m.ids.View(DataType::kInt32, Shape({tokens})));
  INFERX_ASSIGN_OR_RETURN(const TensorView pos_v,
                          m.positions.View(DataType::kInt32, Shape({tokens})));
  INFERX_ASSIGN_OR_RETURN(const TensorView x,
                          m.hidden.View(kBf16, Shape({tokens, h})));
  INFERX_ASSIGN_OR_RETURN(const TensorView resid,
                          m.residual.View(kBf16, Shape({tokens, h})));
  INFERX_ASSIGN_OR_RETURN(const TensorView normed,
                          m.normed.View(kBf16, Shape({tokens, h})));
  INFERX_ASSIGN_OR_RETURN(
      const TensorView q, m.qkv_q.View(kBf16, Shape({tokens, heads, hd})));
  INFERX_ASSIGN_OR_RETURN(
      const TensorView k, m.qkv_k.View(kBf16, Shape({tokens, kv_heads, hd})));
  INFERX_ASSIGN_OR_RETURN(
      const TensorView v, m.qkv_v.View(kBf16, Shape({tokens, kv_heads, hd})));
  INFERX_ASSIGN_OR_RETURN(
      const TensorView attn, m.attn_out.View(kBf16, Shape({tokens, heads, hd})));
  INFERX_ASSIGN_OR_RETURN(
      const TensorView lse_v,
      m.lse.View(DataType::kFloat, Shape({tokens, heads})));
  INFERX_ASSIGN_OR_RETURN(const TensorView proj,
                          m.proj.View(kBf16, Shape({tokens, h})));
  INFERX_ASSIGN_OR_RETURN(const TensorView moe_out,
                          m.moe_out.View(kBf16, Shape({tokens, h})));
  INFERX_ASSIGN_OR_RETURN(
      const TensorView inv_freq,
      m.inv_freq.View(DataType::kFloat, Shape({hd / 2})));

  // Flat views of q/k/v for the projections, which are 2-D.
  INFERX_ASSIGN_OR_RETURN(const TensorView q2,
                          q.Reshape(Shape({tokens, c.q_dim()})));
  INFERX_ASSIGN_OR_RETURN(const TensorView k2,
                          k.Reshape(Shape({tokens, c.kv_dim()})));
  INFERX_ASSIGN_OR_RETURN(const TensorView v2,
                          v.Reshape(Shape({tokens, c.kv_dim()})));
  INFERX_ASSIGN_OR_RETURN(const TensorView attn2,
                          attn.Reshape(Shape({tokens, heads * hd})));

  INFERX_RETURN_IF_ERROR(kernels::EmbeddingLookup(m.embed, ids_v, x));

  // Debugging aid, off unless asked: dump the hidden state after every layer so
  // it can be diffed against HuggingFace's `output_hidden_states`. Localizing a
  // disagreement to a layer is the difference between reading four kernels and
  // reading forty.
  const char* dump_path = std::getenv("INFERX_GPTOSS_DUMP");
  std::FILE* dump = dump_path != nullptr ? std::fopen(dump_path, "wb") : nullptr;

  const char* prof_env = std::getenv("INFERX_GPTOSS_PROFILE");
  ForwardProfile prof;
  const bool profiling = prof_env != nullptr;

  auto dump_hidden = [&](const TensorView& t) {
    if (dump == nullptr) return;
    std::vector<uint16_t> host(static_cast<size_t>(tokens * h));
    (void)cudaMemcpy(host.data(), t.Data(), host.size() * sizeof(uint16_t),
                     cudaMemcpyDeviceToHost);
    std::fwrite(host.data(), sizeof(uint16_t), host.size(), dump);
  };

  dump_hidden(x);

  const float scale = 1.0f / std::sqrt(static_cast<float>(hd));

  for (int64_t layer = 0; layer < c.num_hidden_layers; ++layer) {
    const LayerWeights& w = m.layers[static_cast<size_t>(layer)];

    // --- attention ---------------------------------------------------------
    if (profiling) prof.last = ForwardProfile::Clock::now();
    INFERX_CUDA_RETURN_IF_ERROR(
        cudaMemcpy(resid.Data(), x.Data(), DataTypeByteSize(kBf16, tokens * h),
                   cudaMemcpyDeviceToDevice));

    INFERX_RETURN_IF_ERROR(kernels::RmsNorm(x, w.input_norm, normed,
                                            static_cast<float>(c.rms_norm_eps)));

    // Three separate projections rather than a fused QKV: the checkpoint
    // stores them apart, and this path is not chasing launches.
    INFERX_RETURN_IF_ERROR(m.gemm.LinearBF16(normed, w.q_w, q2));
    INFERX_RETURN_IF_ERROR(kernels::AddBiasInPlace(q2, w.q_b));
    INFERX_RETURN_IF_ERROR(m.gemm.LinearBF16(normed, w.k_w, k2));
    INFERX_RETURN_IF_ERROR(kernels::AddBiasInPlace(k2, w.k_b));
    INFERX_RETURN_IF_ERROR(m.gemm.LinearBF16(normed, w.v_w, v2));
    INFERX_RETURN_IF_ERROR(kernels::AddBiasInPlace(v2, w.v_b));

    INFERX_RETURN_IF_ERROR(kernels::RotaryEmbeddingFromTable(
        q, k, pos_v, inv_freq, m.attn_factor));

    const int64_t window = c.IsSlidingLayer(layer) ? c.sliding_window : 0;

    INFERX_RETURN_IF_ERROR(
        kernels::GptOssAttentionRef(q, k, v, attn, lse_v, window, scale));

    // The sink, folded in afterwards using the lse the kernel just wrote.
    // \see kernels/gpt_oss.h for why this is a rescale rather than a change
    // inside the softmax.
    INFERX_RETURN_IF_ERROR(
        kernels::ApplyAttentionSinks(attn, lse_v, w.sinks, /*lse_is_log2=*/true));

    INFERX_RETURN_IF_ERROR(m.gemm.LinearBF16(attn2, w.o_w, proj));
    INFERX_RETURN_IF_ERROR(kernels::AddBiasInPlace(proj, w.o_b));
    dump_hidden(proj);   // attention output, after o_proj, before the residual
    INFERX_RETURN_IF_ERROR(kernels::AddInPlace(proj, resid));
    INFERX_CUDA_RETURN_IF_ERROR(
        cudaMemcpy(x.Data(), proj.Data(),
                   DataTypeByteSize(kBf16, tokens * h),
                   cudaMemcpyDeviceToDevice));
    if (profiling) prof.tick(prof.attn);

    // --- MoE ---------------------------------------------------------------
    INFERX_CUDA_RETURN_IF_ERROR(
        cudaMemcpy(resid.Data(), x.Data(), DataTypeByteSize(kBf16, tokens * h),
                   cudaMemcpyDeviceToDevice));

    INFERX_RETURN_IF_ERROR(kernels::RmsNorm(x, w.post_norm, normed,
                                            static_cast<float>(c.rms_norm_eps)));

    MoeWeights mw;
    mw.router = w.router_w;
    mw.router_bias = w.router_b;

    INFERX_RETURN_IF_ERROR(
        m.DequantizeLayerExperts(w, &mw.gate_up, &mw.down));
    if (profiling) prof.tick(prof.dequant);

    // The biases are device-resident (uploaded once at Load, de-interleaved
    // for gate_up so it lines up with the split weight). No per-call upload,
    // no bias_keep free, and therefore no per-layer cudaDeviceSynchronize --
    // that sync existed only to keep bias_keep alive until the kernels
    // finished, and with bias_keep gone it is just serialization cost.
    mw.gate_up_bias = w.gate_up_bias_dev;
    mw.down_bias = w.down_bias_dev;

    INFERX_RETURN_IF_ERROR(
        m.moe->Forward(normed, mw, moe_out, &m.gemm));
    if (profiling) prof.tick(prof.moe_forward);

    INFERX_RETURN_IF_ERROR(kernels::AddInPlace(moe_out, resid));
    INFERX_CUDA_RETURN_IF_ERROR(
        cudaMemcpy(x.Data(), moe_out.Data(),
                   DataTypeByteSize(kBf16, tokens * h),
                   cudaMemcpyDeviceToDevice));
    if (profiling) { ++prof.layers; }

    dump_hidden(x);
  }

  if (profiling) prof.report(tokens);

  if (dump != nullptr) std::fclose(dump);

  INFERX_RETURN_IF_ERROR(kernels::RmsNorm(x, m.final_norm, normed,
                                          static_cast<float>(c.rms_norm_eps)));

  INFERX_ASSIGN_OR_RETURN(
      const TensorView logits,
      m.logits.View(kBf16, Shape({tokens, c.vocab_size})));
  INFERX_RETURN_IF_ERROR(m.gemm.LinearBF16(normed, m.lm_head, logits));

  INFERX_CUDA_RETURN_IF_ERROR(cudaDeviceSynchronize());

  // Down to the host as bf16, widened here: a fp32 logits buffer for a 201088
  // vocabulary is 804 KB per position on a card with ~2 GB spare.
  std::vector<uint16_t> raw(static_cast<size_t>(tokens * c.vocab_size));
  INFERX_CUDA_RETURN_IF_ERROR(cudaMemcpy(raw.data(), logits.Data(),
                                         raw.size() * sizeof(uint16_t),
                                         cudaMemcpyDeviceToHost));

  out_logits->resize(raw.size());
  for (size_t i = 0; i < raw.size(); ++i) {
    const uint32_t bits = static_cast<uint32_t>(raw[i]) << 16;
    float value;
    std::memcpy(&value, &bits, sizeof(value));
    (*out_logits)[i] = value;
  }

  return OkStatus();
}

Status GptOssModel::AttachKvCache(int64_t num_blocks, int64_t block_size) {
  KvLayout layout;
  layout.entries_per_token = 2;  // K and V
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

KvBlockPool* GptOssModel::kv_pool() { return impl_->pool.get(); }

Status GptOssModel::ReserveActivations(int64_t max_tokens) {
  if (max_tokens <= 0) {
    return InvalidArgumentError("max_tokens must be positive, got ", max_tokens);
  }
  return impl_->EnsureCapacity(max_tokens);
}

Status GptOssModel::Step(const ForwardBatch& batch,
                         std::vector<float>* out_logits) {
  if (impl_->pool == nullptr) {
    return FailedPreconditionError(
        "Step requires a KV cache; call AttachKvCache first");
  }

  const int64_t total_slots =
      impl_->pool->num_blocks() * impl_->pool->block_size();
  INFERX_RETURN_IF_ERROR(batch.Validate(impl_->config.vocab_size, total_slots));

  INFERX_RETURN_IF_ERROR(impl_->PrepareBatchInputs(batch));
  INFERX_RETURN_IF_ERROR(impl_->RunPagedForward(batch));

  // Download only the rows a caller asked for. At a 201088-wide vocabulary that
  // is 804 KB per row, which is the same reason Qwen2's Step downloads logits
  // indices rather than the whole batch.
  const int64_t vocab = impl_->config.vocab_size;
  out_logits->clear();
  out_logits->reserve(batch.logits_indices.size() *
                      static_cast<size_t>(vocab));

  std::vector<uint16_t> raw(static_cast<size_t>(vocab));
  for (const int32_t index : batch.logits_indices) {
    INFERX_CUDA_RETURN_IF_ERROR(cudaMemcpy(
        raw.data(),
        static_cast<const std::byte*>(impl_->logits.buf.data()) +
            static_cast<size_t>(index) * static_cast<size_t>(vocab) * 2,
        raw.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost));

    const size_t base = out_logits->size();
    out_logits->resize(base + raw.size());
    for (size_t i = 0; i < raw.size(); ++i) {
      const uint32_t bits = static_cast<uint32_t>(raw[i]) << 16;
      float value;
      std::memcpy(&value, &bits, sizeof(value));
      (*out_logits)[base + i] = value;
    }
  }

  return OkStatus();
}

}  // namespace inferx::model
