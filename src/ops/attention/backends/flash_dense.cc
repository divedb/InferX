#include "inferx/ops/attention/backends/flash_dense.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "absl/types/span.h"
#include "inferx/core/device.h"
#include "inferx/core/shape.h"
#include "inferx/ops/attention.h"
#include "inferx/ops/attention/builder.h"
#include "inferx/ops/flash_attention.h"

namespace inferx::attn {
namespace {

using ops::AttentionParams;

/// Non-owning [rows, cols] tensor over device memory: the legacy wrappers
/// consume contiguous rank-2 activations.
Tensor Flat2D(const void* data, std::int64_t rows, std::int64_t cols, DataType dtype,
              Device device) {
  // FromBlob cannot fail on a valid pointer and a fresh shape.
  return std::move(*Tensor::FromBlob(const_cast<void*>(data), dtype, Shape({rows, cols}),
                                     device));
}

/// Non-owning rank-1 int32 tensor over a metadata array (device memory).
Tensor Flat1D(const std::int32_t* data, std::int64_t count, Device device) {
  return std::move(*Tensor::FromBlob(const_cast<std::int32_t*>(data), DataType::kInt32,
                                     Shape({count}), device));
}

/// One dense layer over the FlashInfer paged kernels: separate KV write
/// (ops::WritePagedKv) followed by the plan-workspace attention dispatch.
class FlashDenseKernel final : public Kernel {
 public:
  FlashDenseKernel(AttentionSpec spec, AttentionParams params, Tensor key_cache,
                   Tensor value_cache, std::int64_t block_size, std::int64_t max_tokens)
      : spec_(std::move(spec)),
        params_(params),
        key_cache_(std::move(key_cache)),
        value_cache_(std::move(value_cache)),
        block_size_(block_size),
        max_tokens_(max_tokens) {}

  [[nodiscard]] Status Run(const RunArgs& args) const noexcept override {
    if (args.in.q.rank != 3) {
      return InvalidArgumentError("flash kernel: q must be rank 3 [tokens, heads, dim]");
    }
    const std::int64_t tokens = args.in.q.Dim(0);
    const auto* dense = std::get_if<DenseParams>(&spec_.params);
    const std::int64_t q_width = dense->num_q_heads * dense->head_dim;
    const std::int64_t kv_width = dense->num_kv_heads * dense->head_dim;
    const Device device = args.ctx.GetDevice();

    // The wrappers take contiguous rank-2 activations; fused-QKV strided
    // views must be packed by the caller (declared via
    // needs_unit_stride_head_dim).
    Tensor q = Flat2D(args.in.q.data, tokens, q_width, spec_.q_dtype, device);
    Tensor out = Flat2D(args.out.data, tokens, q_width, spec_.q_dtype, device);

    if (std::getenv("INFERX_DUMP_META") && args.meta.num_reqs > 0) {
      std::vector<int32_t> qo(args.meta.num_reqs + 1), ki(args.meta.num_reqs + 1), lp(args.meta.num_reqs), ib(args.meta.num_tokens);
      auto& mem = args.ctx.Runtime().Memory();
      mem.Copy(qo.data(), args.meta.qo_indptr, qo.size() * 4, CopyKind::kDeviceToHost);
      mem.Copy(ki.data(), args.meta.kv_indptr, ki.size() * 4, CopyKind::kDeviceToHost);
      mem.Copy(lp.data(), args.meta.last_page_len, lp.size() * 4, CopyKind::kDeviceToHost);
      mem.Copy(ib.data(), args.meta.batch_indices, ib.size() * 4, CopyKind::kDeviceToHost);
      std::fprintf(stderr, "META qo=");
      for (auto v : qo) std::fprintf(stderr, "%d,", v);
      std::fprintf(stderr, " kvi=");
      for (auto v : ki) std::fprintf(stderr, "%d,", v);
      std::fprintf(stderr, " lp=");
      for (auto v : lp) std::fprintf(stderr, "%d,", v);
      std::fprintf(stderr, " bi=");
      for (size_t i = 0; i < ib.size() && i < 80; ++i) std::fprintf(stderr, "%d,", ib[i]);
      std::fprintf(stderr, "\n");
    }
    Tensor positions = Flat1D(args.meta.positions, args.meta.num_tokens, device);
    Tensor batch_indices = Flat1D(args.meta.batch_indices, args.meta.num_tokens, device);
    Tensor qo_indptr = Flat1D(args.meta.qo_indptr, args.meta.num_reqs + 1, device);
    Tensor kv_indptr = Flat1D(args.meta.kv_indptr, args.meta.num_reqs + 1, device);
    // Device arrays must never be dereferenced on the host (P6/P10): the
    // page count comes from the batch's host mirror instead.
    std::int64_t pages = 0;
    for (std::int32_t r = 0; r < args.meta.num_reqs; ++r) {
      pages += (args.desc->seq_lens[static_cast<size_t>(r)] + block_size_ - 1) / block_size_;
    }
    Tensor kv_indices = Flat1D(args.meta.kv_indices, pages, device);
    Tensor last_page_len = Flat1D(args.meta.last_page_len, args.meta.num_reqs, device);

    if (args.in.k.Defined()) {
      Tensor k = Flat2D(args.in.k.data, tokens, kv_width, spec_.q_dtype, device);
      Tensor v = Flat2D(args.in.v.data, tokens, kv_width, spec_.q_dtype, device);
      INFERX_RETURN_IF_ERROR(ops::WritePagedKv(args.ctx, k, v, positions, batch_indices,
                                               kv_indptr, kv_indices, key_cache_,
                                               value_cache_, block_size_,
                                               params_.page_stride_elements));
    }

    // The wrappers' contract: the plan buffer is caller-sized once
    // ([3 * max_tokens + 1], docs on AttentionPlanWorkspace) and each step
    // begins with BeginAttentionStep, which flags the plan for rebuild.
    if (!ws_.plan.has_value()) {
      INFERX_ASSIGN_OR_RETURN(auto plan, Tensor::Empty(DataType::kInt32,
                                                        Shape({3 * max_tokens_ + 1}), device));
      ws_.plan = std::move(plan);
    }
    INFERX_RETURN_IF_ERROR(ops::BeginAttentionStep(args.ctx, kv_indptr, last_page_len,
                                                   block_size_, tokens, args.meta.num_reqs,
                                                   ws_));

    // The FlashInfer plan is rebuilt from host mirrors of the query
    // extents; the BatchDesc carried on the step provides them.
    if (args.desc == nullptr) {
      return InternalError("flash kernel needs the step's BatchDesc host mirror");
    }
    host_qo_.assign(args.desc->query_lens.begin(),
                    args.desc->query_lens.begin() + args.meta.num_reqs);
    host_qo_scratch_.assign(host_qo_.begin(), host_qo_.end());
    // PagedAttention takes qo_indptr (prefix sums); build it host-side.
    host_qo_.clear();
    host_qo_.push_back(0);
    std::int32_t running = 0;
    for (std::int32_t len : host_qo_scratch_) {
      running += len;
      host_qo_.push_back(running);
    }

    return ops::PagedAttention(args.ctx, q, qo_indptr, kv_indptr, kv_indices,
                               last_page_len, absl::MakeConstSpan(host_qo_),
                               args.meta.num_reqs, key_cache_, value_cache_, block_size_,
                               params_, ws_, out);
  }

  [[nodiscard]] KvUpdateMode GetKvUpdateMode() const noexcept override {
    return KvUpdateMode::kSeparate;
  }
  [[nodiscard]] std::size_t WorkspaceBytes(const EngineLimits&) const noexcept override {
    return 0;  // The ops wrappers own their plan and split-decode buffers.
  }

 private:
  AttentionSpec spec_;
  AttentionParams params_;
  Tensor key_cache_;
  Tensor value_cache_;
  std::int64_t block_size_;
  std::int64_t max_tokens_;
  /// Plan state shared across steps; the wrappers mutate it lazily.
  mutable ops::AttentionPlanWorkspace ws_;
  mutable std::vector<std::int32_t> host_qo_;
  mutable std::vector<std::int32_t> host_qo_scratch_;
};

constexpr Capability kFlashDenseCapability{
    .name = "flash_dense",
    .mechanisms = FlashDenseBackend::kMechanisms,
    .roles = {Role::kDecoderSelf},
    .masks = {MaskKind::kCausal, MaskKind::kBidirectional, MaskKind::kSlidingWindow},
    .q_dtypes = {DataType::kBFloat16},
    .kv_dtypes = {DataType::kBFloat16},
    .layouts = EnumSet<KvPageLayout>{KvPageLayout::kKvPlanarNhd},
    .phases = {.decode_only = true, .prefill_only = true, .mixed = true},
    .kv_update = KvUpdateMode::kSeparate,
    .graph = GraphSupport::kNever,
    .min_sm = 70,
};

}  // namespace

const Capability& FlashDenseBackend::GetCapability() const noexcept {
  return kFlashDenseCapability;
}

StatusOr<std::unique_ptr<Kernel>> FlashDenseBackend::CreateKernel(
    const AttentionSpec& spec, const CacheView& view, const EngineLimits& limits) const {
  const auto* dense = std::get_if<DenseParams>(&spec.params);
  if (dense == nullptr) {
    return InvalidArgumentError("flash_dense implements the dense family only");
  }
  if (limits.device.Type() != DeviceType::kCuda) {
    return UnimplementedError("flash_dense runs on CUDA devices only");
  }
  if (view.base == nullptr) {
    return InvalidArgumentError("flash_dense needs a bound token-wise cache view");
  }

  AttentionParams params;
  params.query_heads = dense->num_q_heads;
  params.kv_heads = dense->num_kv_heads;
  params.head_dim = dense->head_dim;
  params.scale = spec.softmax_scale != 0.0f
                     ? spec.softmax_scale
                     : 1.0f / std::sqrt(static_cast<float>(dense->head_dim));
  params.sliding_window =
      spec.mask.kind == MaskKind::kSlidingWindow ? spec.mask.window : 0;
  params.softcap = spec.features.Has(Feature::kSoftCap) ? spec.softcap : 0.0f;

  // The pool region is [pages + 1, block_tokens, bytes_per_token] uint8 in
  // kKvPlanarNhd order: the K plane then the V plane, [Hkv, D] per token.
  // Present the two planes as the rank-4 caches the wrappers expect.
  const std::int64_t rows = view.num_pages + 1;
  const std::int64_t plane_bytes =
      view.block_tokens * dense->num_kv_heads * dense->head_dim * 2 /* bf16 */;
  if (view.page_stride_bytes < plane_bytes * 2 || view.page_stride_bytes % 2 != 0) {
    return InvalidArgumentError("flash_dense: invalid KV page stride");
  }
  params.page_stride_elements = view.page_stride_bytes / 2;
  if (view.page_bytes != plane_bytes * 2) {
    return InvalidArgumentError("flash_dense: cache view page bytes ", view.page_bytes,
                                " disagree with the spec geometry ", plane_bytes * 2);
  }
  Device device = limits.device;
  auto key = Tensor::FromBlob(view.base, DataType::kBFloat16,
                              Shape({rows, view.block_tokens, dense->num_kv_heads,
                                     dense->head_dim}),
                              device);
  INFERX_RETURN_IF_ERROR(key.status());
  auto value = Tensor::FromBlob(view.base + plane_bytes, DataType::kBFloat16,
                                Shape({rows, view.block_tokens, dense->num_kv_heads,
                                       dense->head_dim}),
                                device);
  INFERX_RETURN_IF_ERROR(value.status());

  return std::make_unique<FlashDenseKernel>(spec, params, std::move(*key),
                                            std::move(*value), view.block_tokens,
                                            limits.max_tokens);
}

StatusOr<std::unique_ptr<MetadataBuilder>> FlashDenseBackend::CreateBuilder(
    const AttentionSpec&, const EngineLimits&) const {
  return MakeCommonBuilder(GraphSupport::kNever);
}

}  // namespace inferx::attn
