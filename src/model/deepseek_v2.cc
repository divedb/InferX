#include "inferx/model/deepseek_v2.h"

#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/str_cat.h"
#include "inferx/core/device_buffer.h"
#include "inferx/core/device_runtime.h"
#include "inferx/core/tensor.h"
#include "inferx/kernels/gemm.h"
#include "inferx/kernels/layers.h"
#include "inferx/model/mla.h"
#include "inferx/model/moe_ffn.h"
#include "inferx/model/safetensors.h"
#include "inferx/model/weight_loader.h"
#include "inferx/support/log.h"

namespace inferx::model {
namespace {

constexpr DataType kBf16 = DataType::kBFloat16;

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

// The checkpoint stores rope output channels interleaved -- storage pairs
// (2j, 2j+1) -- while our kernel rotates half-split pairs (j, j+half). The
// fix is this load-time gather of the rope rows, even source rows first,
// then odd, applied identically to the Q heads and the shared key so every
// dot product pairs permuted-with-permuted. Measured against HF on real
// weights (docs/DSV2_VALIDATION.md, 2026-08-09): with the permutation, the
// post-rope q/k tensors match HF's to bf16 rounding, settling §18.4 D2 and
// ARCHITECTURE.md's "unverified" caveat.
//
// Identity over `rows`, with each listed `rope`-wide block at `starts`
// replaced by the de-interleaving gather: evens first, then odds.
std::vector<int64_t> DeinterleaveMap(int64_t rows,
                                     const std::vector<int64_t>& starts,
                                     int64_t rope) {
  std::vector<int64_t> map(static_cast<size_t>(rows));
  for (int64_t i = 0; i < rows; ++i) map[static_cast<size_t>(i)] = i;

  const int64_t half = rope / 2;
  for (const int64_t start : starts) {
    for (int64_t j = 0; j < half; ++j) {
      map[static_cast<size_t>(start + j)] = start + 2 * j;
      map[static_cast<size_t>(start + half + j)] = start + 2 * j + 1;
    }
  }
  return map;
}

// One layer's device-resident weights. The FFN half is either dense or MoE,
// never both -- first_k_dense_replace decides which fields are defined.
struct LayerWeights {
  TensorView input_norm;
  TensorView post_norm;

  MlaWeights mla;

  // Dense FFN (layers below first_k_dense_replace): [2·inter, hidden] fused
  // gate|up, and [hidden, inter] down.
  TensorView dense_gate_up;
  TensorView dense_down;

  // MoE FFN: the router plus stacked expert tensors in MoeWeights' layout,
  // and the fused shared-expert MLP. shared_gate stays undefined -- DeepSeek's
  // shared experts are ungated, and MoeFfn rejects a gate under that config.
  TensorView router;
  TensorView experts_gate_up;  // [E, 2·moe_inter, hidden]
  TensorView experts_down;     // [E, hidden, moe_inter]
  TensorView shared_gate_up;   // [2·shared_inter, hidden]
  TensorView shared_down;      // [hidden, shared_inter]
};

// A batch sliced into its per-sequence runs. MLA attention is per sequence
// (the reconstruction GEMM's m is that sequence's context), so the flat batch
// is walked as ranges; the scheduler already emits tokens grouped by sequence
// and PrepareBatchInputs verifies that rather than assuming it.
struct SeqRange {
  int64_t start = 0;    // first token index in the flat batch
  int64_t count = 0;    // tokens this step
  int64_t context = 0;  // cache length after this step's append
  int64_t seq = 0;      // row in the batch's block table
};

}  // namespace

struct DeepseekV2Model::Impl {
  Impl(ModelConfig c, kernels::CublasLtGemm g)
      : config(std::move(c)), gemm(std::move(g)) {}
  ~Impl() {
    if (runtime != nullptr && stream.handle != nullptr) {
      (void)runtime->DestroyStream(stream);
    }
  }

  ModelConfig config;

  std::vector<DeviceBuffer> weight_buffers;
  std::vector<LayerWeights> layers;

  TensorView embed;
  TensorView final_norm;
  TensorView lm_head;

  kernels::CublasLtGemm gemm;
  std::unique_ptr<MlaAttentionLayer> mla;
  std::unique_ptr<MoeFfn> moe;
  DeviceRuntime* runtime = nullptr;
  Stream stream;

  std::unique_ptr<KvBlockPool> pool;

  // Per-step index buffers and the batch's per-sequence decomposition.
  DeviceBuffer slots_buf, block_table_buf;
  int64_t block_table_capacity = 0;
  std::vector<SeqRange> ranges;

  int64_t capacity_tokens = 0;
  Scratch ids, positions;
  Scratch hidden, residual, normed;
  Scratch attn_out, ffn_out;
  Scratch dense_gate_up, dense_act;
  Scratch logits;

  Status EnsureCapacity(int64_t tokens);
  Status PrepareBatchInputs(const ForwardBatch& batch);
  Status RunPagedForward(const ForwardBatch& batch);
};

Status DeepseekV2Model::Impl::EnsureCapacity(int64_t tokens) {
  if (tokens <= capacity_tokens) return OkStatus();

  const size_t sz = DataTypeByteSize(kBf16, 1);
  const int64_t h = config.hidden_size;

  // The widest dense FFN in the stack: the first_k_dense layers use the full
  // intermediate_size, which for V2-Lite (10944) is far wider than one
  // expert's 1408.
  const int64_t dense_inter = config.intermediate_size;

  INFERX_RETURN_IF_ERROR(Grow(&ids, sizeof(int32_t) * tokens));
  INFERX_RETURN_IF_ERROR(Grow(&positions, sizeof(int32_t) * tokens));
  INFERX_RETURN_IF_ERROR(Grow(&hidden, sz * tokens * h));
  INFERX_RETURN_IF_ERROR(Grow(&residual, sz * tokens * h));
  INFERX_RETURN_IF_ERROR(Grow(&normed, sz * tokens * h));
  INFERX_RETURN_IF_ERROR(Grow(&attn_out, sz * tokens * h));
  INFERX_RETURN_IF_ERROR(Grow(&ffn_out, sz * tokens * h));
  INFERX_RETURN_IF_ERROR(Grow(&dense_gate_up, sz * tokens * 2 * dense_inter));
  INFERX_RETURN_IF_ERROR(Grow(&dense_act, sz * tokens * dense_inter));
  INFERX_RETURN_IF_ERROR(Grow(&logits, sz * tokens * config.vocab_size));

  capacity_tokens = tokens;
  return OkStatus();
}

Status DeepseekV2Model::Impl::PrepareBatchInputs(const ForwardBatch& batch) {
  const int64_t tokens = batch.num_tokens();

  INFERX_RETURN_IF_ERROR(EnsureCapacity(tokens));

  const int64_t table_elems = batch.num_seqs * batch.max_blocks_per_seq;
  if (table_elems > block_table_capacity) {
    INFERX_ASSIGN_OR_RETURN(
        block_table_buf, DeviceBuffer::Allocate(
                             static_cast<size_t>(table_elems) * sizeof(int32_t),
                             DeviceId::Cuda(0)));
    block_table_capacity = table_elems;
  }

  const size_t tok_bytes = static_cast<size_t>(tokens) * sizeof(int32_t);
  if (slots_buf.size() < tok_bytes) {
    INFERX_ASSIGN_OR_RETURN(
        slots_buf, DeviceBuffer::Allocate(tok_bytes, DeviceId::Cuda(0)));
  }

  INFERX_RETURN_IF_ERROR(runtime->CopyAsync(ids.buf.data(),
                                            batch.token_ids.data(), tok_bytes,
                                            CopyKind::kHostToDevice, stream));
  INFERX_RETURN_IF_ERROR(runtime->CopyAsync(positions.buf.data(),
                                            batch.positions.data(), tok_bytes,
                                            CopyKind::kHostToDevice, stream));
  INFERX_RETURN_IF_ERROR(runtime->CopyAsync(slots_buf.data(),
                                            batch.slots.data(), tok_bytes,
                                            CopyKind::kHostToDevice, stream));
  INFERX_RETURN_IF_ERROR(
      runtime->CopyAsync(block_table_buf.data(), batch.block_table.data(),
                         static_cast<size_t>(table_elems) * sizeof(int32_t),
                         CopyKind::kHostToDevice, stream));

  // Decompose the flat batch into per-sequence runs. The MLA loop needs each
  // sequence's token range and its context length after the append; the
  // scheduler emits tokens grouped by sequence, and a batch that is not --
  // which would silently attend tokens against the wrong cache -- is rejected
  // here instead.
  ranges.clear();
  for (int64_t t = 0; t < tokens; ++t) {
    const int64_t seq = batch.seq_of_token[static_cast<size_t>(t)];
    if (!ranges.empty() && ranges.back().seq == seq) {
      ranges.back().count += 1;
    } else {
      for (const SeqRange& r : ranges) {
        if (r.seq == seq) {
          return InvalidArgumentError(
              "DeepseekV2Model: batch tokens are not grouped by sequence "
              "(sequence ",
              seq, " reappears at token ", t, ")");
        }
      }
      ranges.push_back({t, 1, 0, seq});
    }
  }
  for (SeqRange& r : ranges) {
    r.context = batch.positions[static_cast<size_t>(r.start + r.count - 1)] + 1;
  }

  return OkStatus();
}

Status DeepseekV2Model::Impl::RunPagedForward(const ForwardBatch& batch) {
  Impl& m = *this;
  const ModelConfig& c = m.config;
  const int64_t tokens = batch.num_tokens();
  const int64_t h = c.hidden_size;

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
  INFERX_ASSIGN_OR_RETURN(const TensorView attn,
                          m.attn_out.View(kBf16, Shape({tokens, h})));
  INFERX_ASSIGN_OR_RETURN(const TensorView ffn,
                          m.ffn_out.View(kBf16, Shape({tokens, h})));

  INFERX_RETURN_IF_ERROR(kernels::EmbeddingLookup(m.embed, ids_v, x, stream));

  const float eps = static_cast<float>(c.rms_norm_eps);
  const size_t row_bytes = DataTypeByteSize(kBf16, tokens * h);

  for (int64_t layer = 0; layer < c.num_hidden_layers; ++layer) {
    const LayerWeights& w = m.layers[static_cast<size_t>(layer)];

    // --- MLA attention ------------------------------------------------------
    INFERX_RETURN_IF_ERROR(runtime->CopyAsync(
        resid.Data(), x.Data(), row_bytes, CopyKind::kDeviceToDevice, stream));
    INFERX_RETURN_IF_ERROR(
        kernels::RmsNorm(x, w.input_norm, normed, eps, stream));

    INFERX_ASSIGN_OR_RETURN(const TensorView cache, m.pool->KeyCache(layer));

    // Per sequence: MLA's reconstruction GEMM runs at that sequence's context
    // length, so the ragged batch is a loop until absorption or the
    // FlashInfer MLA wrapper replaces the inner attention (§18.7 D6).
    for (const SeqRange& r : ranges) {
      INFERX_ASSIGN_OR_RETURN(const TensorView x_s,
                              normed.Slice(r.start, r.start + r.count));
      INFERX_ASSIGN_OR_RETURN(const TensorView pos_s,
                              pos_v.Slice(r.start, r.start + r.count));
      INFERX_ASSIGN_OR_RETURN(const TensorView slots_s,
                              slots_v.Slice(r.start, r.start + r.count));
      INFERX_ASSIGN_OR_RETURN(const TensorView out_s,
                              attn.Slice(r.start, r.start + r.count));

      INFERX_ASSIGN_OR_RETURN(const TensorView row_2d,
                              table_v.Slice(r.seq, r.seq + 1));
      INFERX_ASSIGN_OR_RETURN(
          const TensorView row,
          row_2d.Reshape(Shape({batch.max_blocks_per_seq})));

      INFERX_RETURN_IF_ERROR(m.mla->Forward(x_s, pos_s, slots_s, row, r.context,
                                            cache, w.mla, out_s, &m.gemm,
                                            stream));
    }

    INFERX_RETURN_IF_ERROR(kernels::AddInPlace(attn, resid, stream));
    INFERX_RETURN_IF_ERROR(runtime->CopyAsync(
        x.Data(), attn.Data(), row_bytes, CopyKind::kDeviceToDevice, stream));

    // --- FFN: dense below first_k_dense_replace, routed after --------------
    INFERX_RETURN_IF_ERROR(runtime->CopyAsync(
        resid.Data(), x.Data(), row_bytes, CopyKind::kDeviceToDevice, stream));
    INFERX_RETURN_IF_ERROR(
        kernels::RmsNorm(x, w.post_norm, normed, eps, stream));

    if (c.IsMoeLayer(layer)) {
      MoeWeights mw;
      mw.router = w.router;
      mw.gate_up = w.experts_gate_up;
      mw.down = w.experts_down;
      mw.shared_gate_up = w.shared_gate_up;
      mw.shared_down = w.shared_down;

      INFERX_RETURN_IF_ERROR(m.moe->Forward(normed, mw, ffn, &m.gemm, stream));
    } else {
      const int64_t inter = c.intermediate_size;
      INFERX_ASSIGN_OR_RETURN(
          const TensorView gu,
          m.dense_gate_up.View(kBf16, Shape({tokens, 2 * inter})));
      INFERX_ASSIGN_OR_RETURN(const TensorView act,
                              m.dense_act.View(kBf16, Shape({tokens, inter})));

      INFERX_RETURN_IF_ERROR(
          m.gemm.LinearBF16(normed, w.dense_gate_up, gu, stream));
      INFERX_RETURN_IF_ERROR(kernels::SiluMulFused(gu, act, stream));
      INFERX_RETURN_IF_ERROR(m.gemm.LinearBF16(act, w.dense_down, ffn, stream));
    }

    INFERX_RETURN_IF_ERROR(kernels::AddInPlace(ffn, resid, stream));
    INFERX_RETURN_IF_ERROR(runtime->CopyAsync(
        x.Data(), ffn.Data(), row_bytes, CopyKind::kDeviceToDevice, stream));
  }

  INFERX_RETURN_IF_ERROR(
      kernels::RmsNorm(x, m.final_norm, normed, eps, stream));

  INFERX_ASSIGN_OR_RETURN(const TensorView logits_v,
                          m.logits.View(kBf16, Shape({tokens, c.vocab_size})));
  INFERX_RETURN_IF_ERROR(
      m.gemm.LinearBF16(normed, m.lm_head, logits_v, stream));

  return OkStatus();
}

StatusOr<DeepseekV2Model> DeepseekV2Model::Load(std::string_view dir) {
  INFERX_ASSIGN_OR_RETURN(ModelConfig config, ModelConfig::FromDirectory(dir));

  if (config.architecture != Architecture::kDeepSeekV2) {
    return InvalidArgumentError("DeepseekV2Model: checkpoint declares ",
                                ArchitectureName(config.architecture));
  }
  if (config.weight_dtype != kBf16) {
    return UnimplementedError("DeepseekV2Model: only bf16 checkpoints, got ",
                              DataTypeName(config.weight_dtype));
  }

  INFERX_ASSIGN_OR_RETURN(Checkpoint ckpt, Checkpoint::Open(dir));
  INFERX_ASSIGN_OR_RETURN(kernels::CublasLtGemm gemm,
                          kernels::CublasLtGemm::Create());

  auto impl = std::make_unique<Impl>(config, std::move(gemm));
  INFERX_ASSIGN_OR_RETURN(impl->runtime, RuntimeFor(DeviceId::Cuda(0)));
  INFERX_ASSIGN_OR_RETURN(impl->stream,
                          impl->runtime->CreateStream(DeviceId::Cuda(0)));

  const int64_t h = config.hidden_size;
  const int64_t heads = config.num_attention_heads;
  const int64_t qk = config.qk_nope_head_dim + config.qk_rope_head_dim;
  const int64_t latent = config.kv_lora_rank;
  const int64_t rope = config.qk_rope_head_dim;
  const int64_t vd = config.v_head_dim;
  const int64_t experts = config.num_experts;
  const int64_t moe_inter = config.moe_intermediate_size;
  const int64_t shared_inter = config.shared_expert_intermediate_size;

  // The shared movement layer: pinned staging ring, parallel host-side
  // copies, chunked device allocation. Views it returns are not readable
  // until Finish()/Release() below — everything loads, then the pipeline
  // drains once.
  INFERX_ASSIGN_OR_RETURN(WeightLoader loader, WeightLoader::Create(&ckpt));

  INFERX_ASSIGN_OR_RETURN(
      impl->embed,
      loader.Load("model.embed_tokens.weight", Shape({config.vocab_size, h})));

  INFERX_ASSIGN_OR_RETURN(impl->final_norm,
                          loader.Load("model.norm.weight", Shape({h})));

  if (config.tie_word_embeddings) {
    impl->lm_head = impl->embed;
  } else {
    INFERX_ASSIGN_OR_RETURN(
        impl->lm_head,
        loader.Load("lm_head.weight", Shape({config.vocab_size, h})));
  }

  impl->layers.resize(static_cast<size_t>(config.num_hidden_layers));

  // A 31 GB load is minutes of silence without progress, and silence is
  // indistinguishable from a hang. One line per layer to stderr, the same
  // channel main.cc's "loading ..." uses; per-layer cost is measured because
  // the MoE layers (hundreds of MB of experts each) dominate and a stall
  // should name its layer.
  const auto load_start = std::chrono::steady_clock::now();
  LOG(INFO) << "deepseek-v2: loading " << ckpt.size() << " tensors, "
            << static_cast<double>(ckpt.TotalBytes()) / 1e9 << " GB, "
            << config.num_hidden_layers
            << " layers (rope convention: deinterleaved, " << loader.threads()
            << " copy threads)";

  for (int64_t i = 0; i < config.num_hidden_layers; ++i) {
    LayerWeights& w = impl->layers[static_cast<size_t>(i)];
    const std::string p = absl::StrCat("model.layers.", i, ".");

    const auto layer_start = std::chrono::steady_clock::now();
    const size_t layer_before = loader.stats().bytes;
    const auto log_layer = [&](const char* kind) {
      const auto now = std::chrono::steady_clock::now();
      VLOG(3) << "deepseek-v2: layer " << (i + 1) << '/'
              << config.num_hidden_layers << " (" << kind << ") "
              << static_cast<double>(loader.stats().bytes - layer_before) / 1e6
              << " MB in "
              << std::chrono::duration_cast<std::chrono::milliseconds>(
                     now - layer_start)
                     .count()
              << " ms, " << static_cast<double>(loader.stats().bytes) / 1e9
              << " GB total";
    };

    auto up = [&](const std::string& name, const Shape& expected,
                  TensorView* out) -> Status {
      INFERX_ASSIGN_OR_RETURN(*out, loader.Load(p + name, expected));
      return OkStatus();
    };

    INFERX_RETURN_IF_ERROR(
        up("input_layernorm.weight", Shape({h}), &w.input_norm));
    INFERX_RETURN_IF_ERROR(
        up("post_attention_layernorm.weight", Shape({h}), &w.post_norm));

    // MLA projections. q_lora_rank 0 (V2-Lite's `null`) means one q_proj and
    // no q_a/q_a_norm -- exactly the branch MlaAttentionLayer takes when
    // MlaWeights.q_a is undefined. The rope output channels of the Q
    // projection and of kv_a's shared key are the two places the
    // de-interleave gather applies (see DeinterleaveMap above).

    // The rope block within each Q head's [nope | rope] output rows.
    std::vector<int64_t> q_rope_starts;
    for (int64_t head = 0; head < heads; ++head) {
      q_rope_starts.push_back(head * qk + config.qk_nope_head_dim);
    }

    if (config.q_lora_rank > 0) {
      INFERX_RETURN_IF_ERROR(up("self_attn.q_a_proj.weight",
                                Shape({config.q_lora_rank, h}), &w.mla.q_a));
      INFERX_RETURN_IF_ERROR(up("self_attn.q_a_layernorm.weight",
                                Shape({config.q_lora_rank}), &w.mla.q_a_norm));
      INFERX_ASSIGN_OR_RETURN(
          w.mla.q_b, loader.LoadRowPermuted(
                         p + "self_attn.q_b_proj.weight",
                         Shape({heads * qk, config.q_lora_rank}),
                         DeinterleaveMap(heads * qk, q_rope_starts, rope)));
    } else {
      INFERX_ASSIGN_OR_RETURN(
          w.mla.q_b, loader.LoadRowPermuted(
                         p + "self_attn.q_proj.weight", Shape({heads * qk, h}),
                         DeinterleaveMap(heads * qk, q_rope_starts, rope)));
    }
    INFERX_ASSIGN_OR_RETURN(
        w.mla.kv_a,
        loader.LoadRowPermuted(p + "self_attn.kv_a_proj_with_mqa.weight",
                               Shape({latent + rope, h}),
                               DeinterleaveMap(latent + rope, {latent}, rope)));
    INFERX_RETURN_IF_ERROR(up("self_attn.kv_a_layernorm.weight",
                              Shape({latent}), &w.mla.kv_a_norm));
    INFERX_RETURN_IF_ERROR(up(
        "self_attn.kv_b_proj.weight",
        Shape({heads * (config.qk_nope_head_dim + vd), latent}), &w.mla.kv_b));
    INFERX_RETURN_IF_ERROR(
        up("self_attn.o_proj.weight", Shape({h, heads * vd}), &w.mla.o));

    if (!config.IsMoeLayer(i)) {
      const int64_t inter = config.intermediate_size;
      const std::vector<std::string> gate_up = {p + "mlp.gate_proj.weight",
                                                p + "mlp.up_proj.weight"};
      INFERX_ASSIGN_OR_RETURN(w.dense_gate_up,
                              loader.LoadStacked(gate_up, Shape({inter, h}),
                                                 Shape({2 * inter, h})));
      INFERX_RETURN_IF_ERROR(
          up("mlp.down_proj.weight", Shape({h, inter}), &w.dense_down));
      log_layer("dense");
      continue;
    }

    INFERX_RETURN_IF_ERROR(
        up("mlp.gate.weight", Shape({experts, h}), &w.router));

    // Routed experts, stacked. gate_up interleaves [gate_e | up_e] per expert
    // exactly as MoeWeights documents; down stacks plainly. Declaring the
    // stack to the loader keeps it one streamed device tensor without the
    // old per-layer host repack.
    {
      std::vector<std::string> names;
      names.reserve(static_cast<size_t>(2 * experts));
      for (int64_t e = 0; e < experts; ++e) {
        const std::string ep = absl::StrCat(p, "mlp.experts.", e, ".");
        names.push_back(ep + "gate_proj.weight");
        names.push_back(ep + "up_proj.weight");
      }
      INFERX_ASSIGN_OR_RETURN(
          w.experts_gate_up,
          loader.LoadStacked(names, Shape({moe_inter, h}),
                             Shape({experts, 2 * moe_inter, h})));
    }
    {
      std::vector<std::string> names;
      names.reserve(static_cast<size_t>(experts));
      for (int64_t e = 0; e < experts; ++e) {
        names.push_back(
            absl::StrCat(p, "mlp.experts.", e, ".down_proj.weight"));
      }
      INFERX_ASSIGN_OR_RETURN(
          w.experts_down, loader.LoadStacked(names, Shape({h, moe_inter}),
                                             Shape({experts, h, moe_inter})));
    }

    // The shared experts, stored as one fused MLP of width
    // n_shared_experts × moe_intermediate_size (2816 for V2-Lite).
    if (shared_inter > 0) {
      const std::vector<std::string> gate_up = {
          p + "mlp.shared_experts.gate_proj.weight",
          p + "mlp.shared_experts.up_proj.weight"};
      INFERX_ASSIGN_OR_RETURN(
          w.shared_gate_up,
          loader.LoadStacked(gate_up, Shape({shared_inter, h}),
                             Shape({2 * shared_inter, h})));
      INFERX_RETURN_IF_ERROR(up("mlp.shared_experts.down_proj.weight",
                                Shape({h, shared_inter}), &w.shared_down));
    }

    log_layer("moe");
  }

  // Everything is enqueued; the views become readable here. The stage/h2d
  // split says where a slow load actually spent its time: stage-bound is the
  // disk (cold page cache), h2d-bound is PCIe (the warm floor).
  INFERX_RETURN_IF_ERROR(loader.Finish());
  const WeightLoaderStats& ls = loader.stats();
  LOG(INFO) << "deepseek-v2: loaded " << static_cast<double>(ls.bytes) / 1e9
            << " GB in "
            << std::chrono::duration_cast<std::chrono::duration<double>>(
                   std::chrono::steady_clock::now() - load_start)
                   .count()
            << " s (stage " << ls.stage_seconds << " s, h2d wait "
            << ls.h2d_wait_seconds << " s)";

  INFERX_ASSIGN_OR_RETURN(impl->weight_buffers, loader.Release());

  INFERX_ASSIGN_OR_RETURN(
      MlaAttentionLayer mla,
      MlaAttentionLayer::Create(config, 1, 1, DeviceId::Cuda(0)));
  impl->mla = std::make_unique<MlaAttentionLayer>(std::move(mla));

  MoeFfn::Config moe_config;
  moe_config.hidden = h;
  moe_config.num_experts = experts;
  moe_config.top_k = config.num_experts_per_tok;
  moe_config.moe_intermediate = moe_inter;
  moe_config.shared_intermediate = shared_inter;
  moe_config.norm_topk_prob = config.norm_topk_prob;
  moe_config.shared_gated = false;  // DeepSeek adds its shared experts plainly
  moe_config.routed_scaling_factor =
      static_cast<float>(config.routed_scaling_factor);
  moe_config.activation = MoeFfn::Activation::kSiluMul;

  INFERX_ASSIGN_OR_RETURN(MoeFfn moe,
                          MoeFfn::Create(moe_config, 1, DeviceId::Cuda(0)));
  impl->moe = std::make_unique<MoeFfn>(std::move(moe));

  return DeepseekV2Model(std::move(impl));
}

DeepseekV2Model::DeepseekV2Model(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}
DeepseekV2Model::~DeepseekV2Model() = default;
DeepseekV2Model::DeepseekV2Model(DeepseekV2Model&&) noexcept = default;
DeepseekV2Model& DeepseekV2Model::operator=(DeepseekV2Model&&) noexcept =
    default;

const ModelConfig& DeepseekV2Model::config() const { return impl_->config; }

int64_t DeepseekV2Model::KvBlockBytes(int64_t block_size) const {
  return KvBlockPool::BlockBytes(impl_->config.num_hidden_layers, block_size,
                                 MlaAttentionLayer::LayoutFor(impl_->config));
}

Status DeepseekV2Model::AttachKvCache(int64_t num_blocks, int64_t block_size) {
  INFERX_ASSIGN_OR_RETURN(
      KvBlockPool pool,
      KvBlockPool::Create(impl_->config.num_hidden_layers, num_blocks,
                          block_size,
                          MlaAttentionLayer::LayoutFor(impl_->config)));

  impl_->pool = std::make_unique<KvBlockPool>(std::move(pool));
  return OkStatus();
}

KvBlockPool* DeepseekV2Model::kv_pool() { return impl_->pool.get(); }

Status DeepseekV2Model::ReserveActivations(int64_t max_tokens) {
  if (max_tokens <= 0) {
    return InvalidArgumentError("max_tokens must be positive, got ",
                                max_tokens);
  }
  return impl_->EnsureCapacity(max_tokens);
}

Status DeepseekV2Model::CaptureDecodeGraph(int64_t num_seqs,
                                           int64_t max_blocks_per_seq) {
  (void)num_seqs;
  (void)max_blocks_per_seq;
  // The unabsorbed MLA path sizes its scratch by context length, which changes
  // every step -- there is no fixed shape to record. Callers skip capture for
  // this model (§18.7 D4); graphs arrive with absorption or the FlashInfer
  // MLA wrapper (D6).
  return UnimplementedError(
      "DeepseekV2Model: decode graphs need a fixed-shape MLA decode path");
}

int64_t DeepseekV2Model::captured_graphs() const { return 0; }

Status DeepseekV2Model::Step(const ForwardBatch& batch,
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
  INFERX_RETURN_IF_ERROR(impl_->runtime->SynchronizeStream(impl_->stream));

  const int64_t vocab = impl_->config.vocab_size;
  out_logits->clear();
  out_logits->reserve(batch.logits_indices.size() * static_cast<size_t>(vocab));

  std::vector<uint16_t> raw(static_cast<size_t>(vocab));
  for (const int32_t index : batch.logits_indices) {
    INFERX_RETURN_IF_ERROR(impl_->runtime->Copy(
        raw.data(),
        static_cast<const std::byte*>(impl_->logits.buf.data()) +
            static_cast<size_t>(index) * static_cast<size_t>(vocab) * 2,
        raw.size() * sizeof(uint16_t), CopyKind::kDeviceToHost));

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

Status DeepseekV2Model::Forward(const std::vector<int32_t>& token_ids,
                                std::vector<float>* out_logits) {
  if (token_ids.empty()) {
    return InvalidArgumentError("DeepseekV2Model: empty prompt");
  }

  const int64_t tokens = static_cast<int64_t>(token_ids.size());
  constexpr int64_t kBlockSize = 16;
  const int64_t blocks = (tokens + kBlockSize - 1) / kBlockSize;

  // A temporary latent pool for this one prompt, swapped in around the paged
  // prefill and restored on every exit path. The decode-equals-prefill pin is
  // what licenses using the paged path as the reference: prefilling through
  // the cache computes the same numbers a cacheless recompute would.
  INFERX_ASSIGN_OR_RETURN(
      KvBlockPool pool,
      KvBlockPool::Create(impl_->config.num_hidden_layers, blocks, kBlockSize,
                          MlaAttentionLayer::LayoutFor(impl_->config)));
  auto temp = std::make_unique<KvBlockPool>(std::move(pool));

  ForwardBatch batch;
  batch.num_seqs = 1;
  batch.max_blocks_per_seq = blocks;
  for (int64_t b = 0; b < blocks; ++b) {
    INFERX_ASSIGN_OR_RETURN(const int32_t block, temp->AllocateBlock());
    batch.block_table.push_back(block);
  }
  for (int64_t t = 0; t < tokens; ++t) {
    batch.token_ids.push_back(token_ids[static_cast<size_t>(t)]);
    batch.positions.push_back(static_cast<int32_t>(t));
    batch.seq_of_token.push_back(0);
    batch.slots.push_back(
        batch.block_table[static_cast<size_t>(t / kBlockSize)] * kBlockSize +
        static_cast<int32_t>(t % kBlockSize));
    batch.logits_indices.push_back(static_cast<int32_t>(t));
  }

  std::swap(impl_->pool, temp);
  const Status status = Step(batch, out_logits);
  std::swap(impl_->pool, temp);

  return status;
}

}  // namespace inferx::model
