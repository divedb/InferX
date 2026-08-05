#include "inferx/model/gpt_oss.h"

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

  // The expert weights stay packed on the host and are read per call.
  Tensor gate_up_blocks, gate_up_scales, gate_up_bias;
  Tensor down_blocks, down_scales, down_bias;
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

  // The MXFP4 staging and the dequantized experts. One layer's worth, reused.
  Scratch packed_blocks, packed_scales;
  Scratch expert_gate_up, expert_down;

  Status EnsureCapacity(int64_t tokens);
  Status DequantizeLayerExperts(const LayerWeights& layer,
                                TensorView* gate_up, TensorView* down);
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

  auto stage_and_decode = [&](const Tensor& blocks, const Tensor& scales,
                              int64_t rows, bool deinterleave,
                              Scratch* dest_scratch,
                              TensorView* out) -> Status {
    const int64_t num_blocks = blocks.dim(2);

    const size_t block_bytes = DataTypeByteSize(
        DataType::kUInt8, experts * rows * num_blocks * kMxfp4BytesPerBlock);
    const size_t scale_bytes =
        DataTypeByteSize(DataType::kUInt8, experts * rows * num_blocks);

    INFERX_RETURN_IF_ERROR(Grow(&packed_blocks, block_bytes));
    INFERX_RETURN_IF_ERROR(Grow(&packed_scales, scale_bytes));

    INFERX_CUDA_RETURN_IF_ERROR(cudaMemcpy(packed_blocks.buf.data(),
                                           blocks.data(), block_bytes,
                                           cudaMemcpyHostToDevice));
    INFERX_CUDA_RETURN_IF_ERROR(cudaMemcpy(packed_scales.buf.data(),
                                           scales.data(), scale_bytes,
                                           cudaMemcpyHostToDevice));

    // Flattened over experts: the decode is per row and does not care where an
    // expert begins, and the de-interleave is within a row pair. Keeping the
    // expert axis would only add a dimension for the kernel to ignore.
    INFERX_ASSIGN_OR_RETURN(
        const TensorView blocks_v,
        packed_blocks.View(DataType::kUInt8,
                           Shape({experts * rows, num_blocks,
                                  kMxfp4BytesPerBlock})));
    INFERX_ASSIGN_OR_RETURN(
        const TensorView scales_v,
        packed_scales.View(DataType::kUInt8,
                           Shape({experts * rows, num_blocks})));

    const int64_t width = num_blocks * kMxfp4ValuesPerBlock;

    INFERX_RETURN_IF_ERROR(Grow(
        dest_scratch, DataTypeByteSize(kBf16, experts * rows * width)));

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

  INFERX_RETURN_IF_ERROR(stage_and_decode(
      layer.gate_up_blocks, layer.gate_up_scales, 2 * inter,
      /*deinterleave=*/true, &expert_gate_up, gate_up));

  INFERX_RETURN_IF_ERROR(stage_and_decode(layer.down_blocks, layer.down_scales,
                                          h, /*deinterleave=*/false,
                                          &expert_down, down));

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

    // The experts stay on the host. `Get` borrows the checkpoint's mapping, so
    // this costs no memory until a page is touched.
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
  }

  // The expert biases have to live on the device, and unlike the weights they
  // are small enough (32 x 5760 and 32 x 2880 bf16) to keep for every layer.
  // Uploaded lazily below, at first use, to keep this function readable.

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

    // The biases are small and are uploaded per layer alongside the weights;
    // keeping all 24 layers' resident would cost 24 x 20 MB for no reason.
    //
    // gate_up's bias must be de-interleaved exactly as its weight was. It is
    // indexed by the same `2·inter` axis, so leaving it in checkpoint order
    // would add every gate's bias to an `up` and vice versa -- a wrong model
    // that still runs, which is the failure mode this whole milestone is
    // organized around.
    std::vector<DeviceBuffer> bias_keep;
    INFERX_ASSIGN_OR_RETURN(mw.gate_up_bias,
                            UploadDeinterleaved(&bias_keep, w.gate_up_bias));
    INFERX_ASSIGN_OR_RETURN(mw.down_bias, Upload(&bias_keep, w.down_bias));

    INFERX_RETURN_IF_ERROR(
        m.moe->Forward(normed, mw, moe_out, &m.gemm));

    INFERX_RETURN_IF_ERROR(kernels::AddInPlace(moe_out, resid));
    INFERX_CUDA_RETURN_IF_ERROR(
        cudaMemcpy(x.Data(), moe_out.Data(),
                   DataTypeByteSize(kBf16, tokens * h),
                   cudaMemcpyDeviceToDevice));

    // bias_keep frees here, which is a device free per layer -- unacceptable
    // in a serving path (§6.1) and irrelevant in a reference that already
    // moves 10 GB over PCIe per call.
    INFERX_CUDA_RETURN_IF_ERROR(cudaDeviceSynchronize());

    dump_hidden(x);
  }

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

}  // namespace inferx::model
