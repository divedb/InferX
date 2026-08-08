#include "inferx/model/mla.h"

#include <cmath>
#include <utility>
#include <vector>

#include <cuda_runtime.h>

#include "inferx/core/cuda_utils.h"
#include "inferx/core/device_buffer.h"
#include "inferx/kernels/gpt_oss.h"
#include "inferx/kernels/layers.h"
#include "inferx/kernels/mla.h"

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

}  // namespace

struct MlaAttentionLayer::Impl {
  ModelConfig config;
  int64_t capacity_tokens = 0;
  int64_t capacity_context = 0;

  // YaRN, resolved once at Create. `inv_freq` is empty for a non-YaRN config,
  // which is also the switch Forward branches on.
  DeviceBuffer inv_freq;         // [qk_rope_head_dim / 2] fp32
  float rope_attn_factor = 1.0f;
  float softmax_scale = 0.0f;

  // Query side, sized by the batch.
  Scratch q_a;      // [tokens, q_lora_rank]
  Scratch q;        // [tokens, heads, qk_nope + qk_rope]
  Scratch q_nope;   // [tokens, heads, qk_nope]
  Scratch q_rope;   // [tokens, heads, qk_rope]
  Scratch kv_a;     // [tokens, kv_lora + qk_rope]
  Scratch latent;   // [tokens, kv_lora]
  Scratch new_rope; // [tokens, qk_rope]
  Scratch attn;     // [tokens, heads, v_head_dim]

  // Context side, sized by how far back attention reaches.
  Scratch gathered;   // [context, kv_lora + qk_rope]
  Scratch ctx_latent; // [context, kv_lora]
  Scratch ctx_rope;   // [context, qk_rope]
  Scratch kv;         // [context, heads, qk_nope + v_head_dim]
  Scratch k_nope;     // [context, heads, qk_nope]
  Scratch v;          // [context, heads, v_head_dim]

  Status EnsureCapacity(int64_t tokens, int64_t context);

  /// Rotates the trailing `rope_dim` of each head — through the YaRN table
  /// when the config has one, the closed form otherwise. Both call sites (Q
  /// heads, shared key) must rotate identically or the cache and the queries
  /// disagree, which is why this is one function rather than two branches.
  Status RotateTail(const TensorView& x, int64_t rope_dim,
                    const TensorView& positions, cudaStream_t stream) {
    if (inv_freq.size() > 0) {
      INFERX_ASSIGN_OR_RETURN(
          const TensorView table,
          TensorView::Create(inv_freq.data(), DataType::kFloat,
                             Shape({rope_dim / 2}), DeviceId::Cuda(0)));
      return kernels::MlaRopeFromTable(x, rope_dim, positions, table,
                                       rope_attn_factor, stream);
    }
    return kernels::MlaRopeInPlace(x, rope_dim, positions,
                                   static_cast<float>(config.rope_theta),
                                   stream);
  }
};

Status MlaAttentionLayer::Impl::EnsureCapacity(int64_t tokens,
                                               int64_t context) {
  const size_t sz = DataTypeByteSize(kBf16, 1);
  const int64_t heads = config.num_attention_heads;
  const int64_t nope = config.qk_nope_head_dim;
  const int64_t rope = config.qk_rope_head_dim;
  const int64_t vd = config.v_head_dim;
  const int64_t latent_dim = config.kv_lora_rank;

  if (tokens > capacity_tokens) {
    if (config.q_lora_rank > 0) {
      INFERX_RETURN_IF_ERROR(Grow(&q_a, sz * tokens * config.q_lora_rank));
    }
    INFERX_RETURN_IF_ERROR(Grow(&q, sz * tokens * heads * (nope + rope)));
    INFERX_RETURN_IF_ERROR(Grow(&q_nope, sz * tokens * heads * nope));
    INFERX_RETURN_IF_ERROR(Grow(&q_rope, sz * tokens * heads * rope));
    INFERX_RETURN_IF_ERROR(Grow(&kv_a, sz * tokens * (latent_dim + rope)));
    INFERX_RETURN_IF_ERROR(Grow(&latent, sz * tokens * latent_dim));
    INFERX_RETURN_IF_ERROR(Grow(&new_rope, sz * tokens * rope));
    INFERX_RETURN_IF_ERROR(Grow(&attn, sz * tokens * heads * vd));
    capacity_tokens = tokens;
  }

  if (context > capacity_context) {
    INFERX_RETURN_IF_ERROR(Grow(&gathered, sz * context * (latent_dim + rope)));
    INFERX_RETURN_IF_ERROR(Grow(&ctx_latent, sz * context * latent_dim));
    INFERX_RETURN_IF_ERROR(Grow(&ctx_rope, sz * context * rope));
    INFERX_RETURN_IF_ERROR(Grow(&kv, sz * context * heads * (nope + vd)));
    INFERX_RETURN_IF_ERROR(Grow(&k_nope, sz * context * heads * nope));
    INFERX_RETURN_IF_ERROR(Grow(&v, sz * context * heads * vd));
    capacity_context = context;
  }

  return OkStatus();
}

KvLayout MlaAttentionLayer::LayoutFor(const ModelConfig& config) {
  KvLayout layout;

  // One entry, one head: the latent is not split into K and V and is not
  // per-head. This is the instantiation §7.3 said `KvLayout` had to admit.
  layout.entries_per_token = 1;
  layout.kv_heads = 1;
  layout.head_dim = config.kv_lora_rank + config.qk_rope_head_dim;
  layout.dtype = DataType::kBFloat16;

  return layout;
}

StatusOr<MlaAttentionLayer> MlaAttentionLayer::Create(const ModelConfig& config,
                                                      int64_t max_tokens,
                                                      int64_t max_context) {
  if (!config.is_mla()) {
    return InvalidArgumentError("MlaAttentionLayer: config is not MLA "
                                "(kv_lora_rank is 0)");
  }

  if (max_tokens <= 0 || max_context <= 0) {
    return InvalidArgumentError("MlaAttentionLayer: max_tokens and max_context "
                                "must be positive, got ", max_tokens, " and ",
                                max_context);
  }

  auto impl = std::make_unique<Impl>();
  impl->config = config;

  // The scale q·k is divided by. HF applies the mscale_all_dim correction to
  // the softmax scale only when the coefficient is set, and squares it: one
  // factor for the query side, one for the key.
  impl->softmax_scale =
      1.0f / std::sqrt(static_cast<float>(config.qk_nope_head_dim +
                                          config.qk_rope_head_dim));

  if (config.is_yarn()) {
    if (config.yarn_mscale_all_dim != 0.0) {
      const float m = kernels::YarnMscale(config.yarn_factor,
                                          config.yarn_mscale_all_dim);
      impl->softmax_scale *= m * m;
    }

    // The rope's own cos/sin factor. Equal coefficients — V2-Lite's case —
    // cancel to exactly 1, moving the whole mscale effect into the softmax
    // scale above.
    impl->rope_attn_factor =
        kernels::YarnMscale(config.yarn_factor, config.yarn_mscale) /
        kernels::YarnMscale(config.yarn_factor, config.yarn_mscale_all_dim);

    // The blended frequency ladder, over the rope sub-vector's own width.
    const int64_t half = config.qk_rope_head_dim / 2;
    std::vector<float> table(static_cast<size_t>(half));
    kernels::ComputeYarnInvFreq(config.qk_rope_head_dim, config.rope_theta,
                                config.yarn_factor, config.yarn_beta_fast,
                                config.yarn_beta_slow,
                                config.yarn_original_max_position,
                                config.yarn_truncate, table.data());

    const size_t bytes = sizeof(float) * table.size();
    INFERX_ASSIGN_OR_RETURN(impl->inv_freq,
                            DeviceBuffer::Allocate(bytes, DeviceId::Cuda(0)));
    INFERX_CUDA_RETURN_IF_ERROR(cudaMemcpy(impl->inv_freq.data(), table.data(),
                                           bytes, cudaMemcpyHostToDevice));
  }

  INFERX_RETURN_IF_ERROR(impl->EnsureCapacity(max_tokens, max_context));

  return MlaAttentionLayer(std::move(impl));
}

float MlaAttentionLayer::softmax_scale() const { return impl_->softmax_scale; }

MlaAttentionLayer::MlaAttentionLayer(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}
MlaAttentionLayer::~MlaAttentionLayer() = default;
MlaAttentionLayer::MlaAttentionLayer(MlaAttentionLayer&&) noexcept = default;
MlaAttentionLayer& MlaAttentionLayer::operator=(MlaAttentionLayer&&) noexcept =
    default;

Status MlaAttentionLayer::Forward(const TensorView& x,
                                  const TensorView& positions,
                                  const TensorView& slot_mapping,
                                  const TensorView& block_table,
                                  int64_t context_len, const TensorView& cache,
                                  const MlaWeights& weights,
                                  const TensorView& out,
                                  kernels::CublasLtGemm* gemm,
                                  cudaStream_t stream) {
  if (gemm == nullptr) return InvalidArgumentError("MLA: gemm is null");

  const ModelConfig& c = impl_->config;
  const int64_t tokens = x.Dim(0);
  const int64_t heads = c.num_attention_heads;
  const int64_t nope = c.qk_nope_head_dim;
  const int64_t rope = c.qk_rope_head_dim;
  const int64_t vd = c.v_head_dim;
  const int64_t latent_dim = c.kv_lora_rank;

  if (x.Rank() != 2 || x.Dim(1) != c.hidden_size) {
    return InvalidArgumentError("MLA: x must be [tokens, ", c.hidden_size,
                                "], got ", x.GetShape().ToString());
  }

  if (context_len < tokens) {
    return InvalidArgumentError("MLA: context_len ", context_len,
                                " is shorter than the ", tokens,
                                " tokens being added to it");
  }

  if (tokens == 0) return OkStatus();

  INFERX_RETURN_IF_ERROR(impl_->EnsureCapacity(tokens, context_len));

  // --- Q: down-project, norm, up-project, then rotate its tail --------------

  INFERX_ASSIGN_OR_RETURN(
      const TensorView q_v,
      impl_->q.View(kBf16, Shape({tokens, heads * (nope + rope)})));

  if (c.q_lora_rank > 0) {
    INFERX_ASSIGN_OR_RETURN(
        const TensorView q_a_v,
        impl_->q_a.View(kBf16, Shape({tokens, c.q_lora_rank})));

    INFERX_RETURN_IF_ERROR(gemm->LinearBF16(x, weights.q_a, q_a_v, stream));
    INFERX_RETURN_IF_ERROR(kernels::RmsNorm(q_a_v, weights.q_a_norm, q_a_v,
                                            static_cast<float>(c.rms_norm_eps),
                                            stream));
    INFERX_RETURN_IF_ERROR(gemm->LinearBF16(q_a_v, weights.q_b, q_v, stream));
  } else {
    INFERX_RETURN_IF_ERROR(gemm->LinearBF16(x, weights.q_b, q_v, stream));
  }

  INFERX_ASSIGN_OR_RETURN(const TensorView q_heads,
                          q_v.Reshape(Shape({tokens, heads, nope + rope})));
  INFERX_RETURN_IF_ERROR(
      impl_->RotateTail(q_heads, rope, positions, stream));

  // Split into the two contiguous halves the attention kernel wants. The
  // rotation happened in place above precisely so this is the only copy.
  INFERX_ASSIGN_OR_RETURN(
      const TensorView q_nope_v,
      impl_->q_nope.View(kBf16, Shape({tokens, heads, nope})));
  INFERX_ASSIGN_OR_RETURN(
      const TensorView q_rope_v,
      impl_->q_rope.View(kBf16, Shape({tokens, heads, rope})));

  INFERX_RETURN_IF_ERROR(kernels::SplitTrailing(q_heads, q_nope_v, q_rope_v,
                                                stream));

  // --- KV: down-project to the latent, norm it, rotate the shared key ------

  INFERX_ASSIGN_OR_RETURN(
      const TensorView kv_a_v,
      impl_->kv_a.View(kBf16, Shape({tokens, latent_dim + rope})));
  INFERX_RETURN_IF_ERROR(gemm->LinearBF16(x, weights.kv_a, kv_a_v, stream));

  INFERX_ASSIGN_OR_RETURN(
      const TensorView latent_v,
      impl_->latent.View(kBf16, Shape({tokens, latent_dim})));
  INFERX_ASSIGN_OR_RETURN(
      const TensorView new_rope_v,
      impl_->new_rope.View(kBf16, Shape({tokens, rope})));

  INFERX_ASSIGN_OR_RETURN(const TensorView kv_a_3d,
                          kv_a_v.Reshape(Shape({tokens, 1, latent_dim + rope})));
  INFERX_ASSIGN_OR_RETURN(const TensorView latent_3d,
                          latent_v.Reshape(Shape({tokens, 1, latent_dim})));
  INFERX_ASSIGN_OR_RETURN(const TensorView new_rope_3d,
                          new_rope_v.Reshape(Shape({tokens, 1, rope})));

  // The RoPE key is rotated as one head shared by every query head, which is
  // the "decoupled" in decoupled RoPE: rotating it per head would cost
  // `heads` times the cache and change nothing.
  INFERX_RETURN_IF_ERROR(
      impl_->RotateTail(kv_a_3d, rope, positions, stream));
  INFERX_RETURN_IF_ERROR(
      kernels::SplitTrailing(kv_a_3d, latent_3d, new_rope_3d, stream));

  // Normalized *before* caching, so a cache hit needs no norm on the way out
  // and the cached bytes are the same bytes attention will use.
  INFERX_RETURN_IF_ERROR(kernels::RmsNorm(latent_v, weights.kv_a_norm, latent_v,
                                          static_cast<float>(c.rms_norm_eps),
                                          stream));

  INFERX_RETURN_IF_ERROR(kernels::MlaAppendLatent(latent_v, new_rope_v, cache,
                                                  slot_mapping, stream));

  // --- Reconstruct K and V for the whole context ---------------------------
  //
  // The unabsorbed form, and the expensive part: every step re-projects every
  // cached token. Absorption removes this entirely (see the class comment).

  INFERX_ASSIGN_OR_RETURN(
      const TensorView gathered_v,
      impl_->gathered.View(kBf16, Shape({context_len, latent_dim + rope})));
  INFERX_RETURN_IF_ERROR(kernels::MlaGatherLatents(cache, block_table,
                                                   context_len, gathered_v,
                                                   stream));

  INFERX_ASSIGN_OR_RETURN(
      const TensorView ctx_latent_v,
      impl_->ctx_latent.View(kBf16, Shape({context_len, latent_dim})));
  INFERX_ASSIGN_OR_RETURN(
      const TensorView ctx_rope_v,
      impl_->ctx_rope.View(kBf16, Shape({context_len, rope})));

  INFERX_ASSIGN_OR_RETURN(
      const TensorView gathered_3d,
      gathered_v.Reshape(Shape({context_len, 1, latent_dim + rope})));
  INFERX_ASSIGN_OR_RETURN(
      const TensorView ctx_latent_3d,
      ctx_latent_v.Reshape(Shape({context_len, 1, latent_dim})));
  INFERX_ASSIGN_OR_RETURN(
      const TensorView ctx_rope_3d,
      ctx_rope_v.Reshape(Shape({context_len, 1, rope})));

  INFERX_RETURN_IF_ERROR(kernels::SplitTrailing(gathered_3d, ctx_latent_3d,
                                                ctx_rope_3d, stream));

  INFERX_ASSIGN_OR_RETURN(
      const TensorView kv_v,
      impl_->kv.View(kBf16, Shape({context_len, heads * (nope + vd)})));
  INFERX_RETURN_IF_ERROR(
      gemm->LinearBF16(ctx_latent_v, weights.kv_b, kv_v, stream));

  INFERX_ASSIGN_OR_RETURN(
      const TensorView kv_heads_v,
      kv_v.Reshape(Shape({context_len, heads, nope + vd})));
  INFERX_ASSIGN_OR_RETURN(
      const TensorView k_nope_v,
      impl_->k_nope.View(kBf16, Shape({context_len, heads, nope})));
  INFERX_ASSIGN_OR_RETURN(
      const TensorView v_v,
      impl_->v.View(kBf16, Shape({context_len, heads, vd})));

  INFERX_RETURN_IF_ERROR(
      kernels::SplitTrailing(kv_heads_v, k_nope_v, v_v, stream));

  // --- Attend, then project out --------------------------------------------

  INFERX_ASSIGN_OR_RETURN(
      const TensorView attn_v,
      impl_->attn.View(kBf16, Shape({tokens, heads, vd})));

  INFERX_RETURN_IF_ERROR(kernels::MlaAttention(
      q_nope_v, q_rope_v, k_nope_v, ctx_rope_v, v_v, attn_v,
      context_len - tokens, impl_->softmax_scale, stream));

  INFERX_ASSIGN_OR_RETURN(const TensorView attn_2d,
                          attn_v.Reshape(Shape({tokens, heads * vd})));
  INFERX_RETURN_IF_ERROR(
      gemm->LinearBF16(attn_2d, weights.o, out, stream));

  return OkStatus();
}

}  // namespace inferx::model
