#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <flashinfer/attention/default_prefill_params.cuh>
#include <flashinfer/attention/mask.cuh>
#include <flashinfer/attention/prefill.cuh>
#include <flashinfer/attention/scheduler.cuh>
#include <flashinfer/attention/variants.cuh>
#include <flashinfer/page.cuh>
#include <flashinfer/pos_enc.cuh>
#include <flashinfer/utils.cuh>
#include <utility>
#include <vector>

#include "inferx/core/cuda_utils.h"
#include "inferx/core/device_buffer.h"
#include "inferx/kernels/flashinfer_prefill.h"

namespace inferx::kernels {
namespace {

using bf16 = __nv_bfloat16;
using IdType = int32_t;

using Params = flashinfer::BatchPrefillPagedParams<bf16, bf16, bf16, IdType>;

// FP8-e4m3 KV: Q/O bf16, K/V fp8, fp32 accumulate. Same dtype-parametric
// templates as decode -- the fa2 prefill upcasts fp8 on load, so this is
// sm_89-capable and not Hopper-gated.
using Fp8 = __nv_fp8_e4m3;
using Fp8Params = flashinfer::BatchPrefillPagedParams<bf16, Fp8, bf16, IdType>;

// No custom mask, no sliding window, no soft cap, no ALiBi -- the same four
// flags the decode path turns off, for the same reasons. Causal masking is a
// MaskMode, not a variant flag, and is selected at dispatch below.
using Variant = flashinfer::DefaultAttention<false, false, false, false>;

Status CheckBf16(const TensorView& t, int rank, const char* name) {
  if (!t.IsDefined()) return InvalidArgumentError(name, " is undefined");
  if (!t.IsCuda()) {
    return InvalidArgumentError(name, " is not on a CUDA device");
  }
  if (t.GetDataType() != DataType::kBFloat16) {
    return InvalidArgumentError(name, " is ", DataTypeName(t.GetDataType()),
                                ", expected bf16");
  }
  if (t.Rank() != rank) {
    return InvalidArgumentError(name, " has rank ", t.Rank(), ", expected ",
                                rank);
  }
  return OkStatus();
}

Status CheckI32(const TensorView& t, const char* name) {
  if (!t.IsDefined()) return InvalidArgumentError(name, " is undefined");
  if (!t.IsCuda()) {
    return InvalidArgumentError(name, " is not on a CUDA device");
  }
  if (t.GetDataType() != DataType::kInt32) {
    return InvalidArgumentError(name, " is ", DataTypeName(t.GetDataType()),
                                ", expected int32");
  }
  return OkStatus();
}

// fp8 KV cache check -- shape rules unchanged from bf16, only the element type.
Status CheckFp8(const TensorView& t, int rank, const char* name) {
  if (!t.IsDefined()) return InvalidArgumentError(name, " is undefined");
  if (!t.IsCuda()) {
    return InvalidArgumentError(name, " is not on a CUDA device");
  }
  if (t.GetDataType() != DataType::kFloat8E4M3FN) {
    return InvalidArgumentError(name, " is ", DataTypeName(t.GetDataType()),
                                ", expected fp8 e4m3");
  }
  if (t.Rank() != rank) {
    return InvalidArgumentError(name, " has rank ", t.Rank(), ", expected ",
                                rank);
  }
  return OkStatus();
}

// V's dequant scale, applied to the output after the kernel -- same place
// FlashInfer's wrapper puts it (and the same TODO: not fused into the kernel).
__global__ void ScaleBf16Kernel(bf16* x, int64_t n, float scale) {
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < n; i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    x[i] = __float2bfloat16(__bfloat162float(x[i]) * scale);
  }
}

// dst = src * scale, out-of-place. Used to fold a dequant scale into the query
// without mutating the caller's tensor.
__global__ void ScaleBf16CopyKernel(const bf16* __restrict__ src,
                                    bf16* __restrict__ dst, int64_t n,
                                    float scale) {
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < n; i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    dst[i] = __float2bfloat16(__bfloat162float(src[i]) * scale);
  }
}

int GridFor(int64_t n) {
  constexpr int kBlock = 256;
  const int64_t want = (n + kBlock - 1) / kBlock;
  return static_cast<int>(want < 1024 ? (want < 1 ? 1 : want) : 1024);
}

}  // namespace

struct FlashInferPrefill::Impl {
  DeviceBuffer float_workspace;
  DeviceBuffer int_workspace;

  flashinfer::PrefillPlanInfo plan;
  bool planned = false;
  int64_t planned_batch = 0;
  int64_t planned_q_heads = 0;
  int64_t planned_kv_heads = 0;
  int64_t planned_head_dim = 0;
  int64_t planned_page_size = 0;
  int64_t planned_total_rows = 0;

  // PrefillPlan stages its integer plan through page-locked memory before the
  // async copy to device, so this cannot be ordinary host memory.
  void* page_locked = nullptr;

  ~Impl() {
    if (page_locked != nullptr) cudaFreeHost(page_locked);
  }
};

StatusOr<std::unique_ptr<FlashInferPrefill>> FlashInferPrefill::Create(
    size_t float_workspace_bytes, size_t int_workspace_bytes) {
  auto impl = std::make_unique<Impl>();

  INFERX_ASSIGN_OR_RETURN(
      impl->float_workspace,
      DeviceBuffer::Allocate(float_workspace_bytes, DeviceId::Cuda(0)));
  INFERX_ASSIGN_OR_RETURN(
      impl->int_workspace,
      DeviceBuffer::Allocate(int_workspace_bytes, DeviceId::Cuda(0)));

  INFERX_CUDA_RETURN_IF_ERROR(cudaHostAlloc(
      &impl->page_locked, int_workspace_bytes, cudaHostAllocDefault));

  return std::unique_ptr<FlashInferPrefill>(
      new FlashInferPrefill(std::move(impl)));
}

FlashInferPrefill::FlashInferPrefill(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}
FlashInferPrefill::~FlashInferPrefill() = default;

Status FlashInferPrefill::Plan(int64_t batch, int64_t q_heads, int64_t kv_heads,
                               int64_t head_dim, int64_t page_size,
                               absl::Span<const int32_t> qo_indptr_host,
                               absl::Span<const int32_t> kv_indptr_host,
                               int64_t total_rows, cudaStream_t stream) {
  if (batch <= 0) {
    return InvalidArgumentError("batch must be positive, got ", batch);
  }
  if (head_dim != 128) {
    return UnimplementedError(
        "only head_dim 128 is compiled; got ", head_dim,
        ". Adding one means instantiating the dispatch for it, which costs "
        "compile time, so it is done on demand rather than speculatively");
  }
  if (kv_heads <= 0 || q_heads % kv_heads != 0) {
    return InvalidArgumentError("q_heads (", q_heads,
                                ") is not a multiple of kv_heads (", kv_heads,
                                ")");
  }
  if (static_cast<int64_t>(qo_indptr_host.size()) != batch + 1) {
    return InvalidArgumentError("qo_indptr has ", qo_indptr_host.size(),
                                " entries, expected ", batch + 1);
  }
  if (static_cast<int64_t>(kv_indptr_host.size()) != batch + 1) {
    return InvalidArgumentError("kv_indptr has ", kv_indptr_host.size(),
                                " entries, expected ", batch + 1);
  }
  if (total_rows != qo_indptr_host.back()) {
    return InvalidArgumentError("total_rows is ", total_rows,
                                " but qo_indptr ends at ",
                                qo_indptr_host.back());
  }

  // Non-const because the planner takes mutable pointers; it does not write
  // through them, but the signature says IdType* and copying is cheaper than
  // arguing with it.
  std::vector<IdType> qo(qo_indptr_host.begin(), qo_indptr_host.end());
  std::vector<IdType> kv(kv_indptr_host.begin(), kv_indptr_host.end());

  const cudaError_t err = flashinfer::PrefillPlan<IdType>(
      impl_->float_workspace.data(), impl_->float_workspace.size(),
      impl_->int_workspace.data(), impl_->page_locked,
      impl_->int_workspace.size(), impl_->plan, qo.data(), kv.data(),
      static_cast<uint32_t>(total_rows), static_cast<uint32_t>(batch),
      static_cast<uint32_t>(q_heads), static_cast<uint32_t>(kv_heads),
      static_cast<uint32_t>(head_dim), static_cast<uint32_t>(head_dim),
      static_cast<uint32_t>(page_size),
      /*enable_cuda_graph=*/false, sizeof(bf16), /*window_left=*/-1,
      /*fixed_split_size=*/-1, /*disable_split_kv=*/false,
      /*num_colocated_ctas=*/0, stream);

  if (err != cudaSuccess) {
    return InternalError("PrefillPlan failed: ", cudaGetErrorString(err));
  }

  impl_->planned = true;
  impl_->planned_batch = batch;
  impl_->planned_q_heads = q_heads;
  impl_->planned_kv_heads = kv_heads;
  impl_->planned_head_dim = head_dim;
  impl_->planned_page_size = page_size;
  impl_->planned_total_rows = total_rows;

  return OkStatus();
}

Status FlashInferPrefill::Run(
    const TensorView& q, const TensorView& k_cache, const TensorView& v_cache,
    const TensorView& qo_indptr, const TensorView& kv_indices,
    const TensorView& kv_indptr, const TensorView& last_page_len,
    const TensorView& out, float scale, cudaStream_t stream) {
  INFERX_RETURN_IF_ERROR(CheckBf16(q, 3, "q"));
  INFERX_RETURN_IF_ERROR(CheckBf16(k_cache, 4, "k_cache"));
  INFERX_RETURN_IF_ERROR(CheckBf16(v_cache, 4, "v_cache"));
  INFERX_RETURN_IF_ERROR(CheckBf16(out, 3, "out"));
  INFERX_RETURN_IF_ERROR(CheckI32(qo_indptr, "qo_indptr"));
  INFERX_RETURN_IF_ERROR(CheckI32(kv_indices, "kv_indices"));
  INFERX_RETURN_IF_ERROR(CheckI32(kv_indptr, "kv_indptr"));
  INFERX_RETURN_IF_ERROR(CheckI32(last_page_len, "last_page_len"));

  if (!impl_->planned) {
    return FailedPreconditionError("Run() called before Plan()");
  }

  const int64_t total_rows = q.Dim(0);
  const int64_t q_heads = q.Dim(1);
  const int64_t head_dim = q.Dim(2);
  const int64_t page_size = k_cache.Dim(1);
  const int64_t kv_heads = k_cache.Dim(2);

  // Checked rather than trusted: a kernel launched with a plan built for
  // another shape reads past the end of its own index arrays.
  if (total_rows != impl_->planned_total_rows ||
      q_heads != impl_->planned_q_heads ||
      kv_heads != impl_->planned_kv_heads ||
      head_dim != impl_->planned_head_dim ||
      page_size != impl_->planned_page_size) {
    return InvalidArgumentError(
        "Run() shape does not match the last Plan(): planned ",
        impl_->planned_total_rows, " rows / ", impl_->planned_q_heads,
        " q heads / ", impl_->planned_kv_heads, " kv heads / head_dim ",
        impl_->planned_head_dim, " / page ", impl_->planned_page_size, ", got ",
        total_rows, " / ", q_heads, " / ", kv_heads, " / ", head_dim, " / ",
        page_size);
  }

  if (out.Dim(0) != total_rows || out.Dim(1) != q_heads ||
      out.Dim(2) != head_dim) {
    return InvalidArgumentError("out is [", out.Dim(0), ",", out.Dim(1), ",",
                                out.Dim(2), "], expected [", total_rows, ",",
                                q_heads, ",", head_dim, "]");
  }

  flashinfer::paged_kv_t<bf16, IdType> paged_kv(
      static_cast<uint32_t>(kv_heads), static_cast<uint32_t>(page_size),
      static_cast<uint32_t>(head_dim),
      static_cast<uint32_t>(impl_->planned_batch), flashinfer::QKVLayout::kNHD,
      static_cast<bf16*>(k_cache.Data()), static_cast<bf16*>(v_cache.Data()),
      static_cast<IdType*>(kv_indices.Data()),
      static_cast<IdType*>(kv_indptr.Data()),
      static_cast<IdType*>(last_page_len.Data()));

  Params params;
  params.q = static_cast<bf16*>(q.Data());
  params.paged_kv = paged_kv;
  params.q_indptr = static_cast<IdType*>(qo_indptr.Data());
  params.maybe_mask_indptr = nullptr;
  params.maybe_q_rope_offset = nullptr;
  params.o = static_cast<bf16*>(out.Data());
  params.lse = nullptr;
  params.maybe_alibi_slopes = nullptr;
  params.num_qo_heads = static_cast<uint32_t>(q_heads);
  // BatchPrefillPagedParams' default constructor leaves this fast divider
  // empty.  The kernel uses it to map packed GQA rows back to query-token
  // indices; with a zero divisor the causal predicate masks every logit.
  params.group_size =
      flashinfer::uint_fastdiv(static_cast<uint32_t>(q_heads / kv_heads));
  params.q_stride_n = static_cast<IdType>(q_heads * head_dim);
  params.q_stride_h = static_cast<IdType>(head_dim);
  params.window_left = -1;
  params.logits_soft_cap = 0.f;
  params.sm_scale = scale;
  params.rope_rcp_scale = 1.f;
  params.rope_rcp_theta = 1.f;
  params.maybe_prefix_len_ptr = nullptr;
  params.token_pos_in_items_len = 0;

  auto* int_base = static_cast<std::byte*>(impl_->int_workspace.data());
  const auto at = [&](size_t offset) {
    return reinterpret_cast<IdType*>(int_base + offset);
  };

  params.request_indices = at(impl_->plan.request_indices_offset);
  params.qo_tile_indices = at(impl_->plan.qo_tile_indices_offset);
  params.kv_tile_indices = at(impl_->plan.kv_tile_indices_offset);
  params.merge_indptr = at(impl_->plan.merge_indptr_offset);
  params.o_indptr = at(impl_->plan.o_indptr_offset);
  params.kv_chunk_size_ptr = at(impl_->plan.kv_chunk_size_ptr_offset);
  params.max_total_num_rows = static_cast<uint32_t>(impl_->plan.total_num_rows);
  params.total_num_rows = nullptr;
  params.padded_batch_size =
      static_cast<uint32_t>(impl_->plan.padded_batch_size);
  params.partition_kv = impl_->plan.split_kv;

  params.block_valid_mask =
      impl_->plan.split_kv ? reinterpret_cast<bool*>(
                                 int_base + impl_->plan.block_valid_mask_offset)
                           : nullptr;

  auto* float_base = static_cast<std::byte*>(impl_->float_workspace.data());

  bf16* tmp_v = nullptr;
  float* tmp_s = nullptr;

  if (impl_->plan.split_kv) {
    tmp_v = reinterpret_cast<bf16*>(float_base + impl_->plan.v_offset);
    tmp_s = reinterpret_cast<float*>(float_base + impl_->plan.s_offset);
  }

  // The planner chooses the query tile, so the dispatch has to follow it rather
  // than pick one: a kernel compiled for a different CTA_TILE_Q than the plan
  // was built with walks the tile indices wrongly.
  cudaError_t err = cudaSuccess;

  DISPATCH_CTA_TILE_Q(impl_->plan.cta_tile_q, CTA_TILE_Q, {
    err = flashinfer::BatchPrefillWithPagedKVCacheDispatched<
        CTA_TILE_Q, /*HEAD_DIM_QK=*/128, /*HEAD_DIM_VO=*/128,
        flashinfer::PosEncodingMode::kNone, /*USE_FP16_QK_REDUCTION=*/false,
        flashinfer::MaskMode::kCausal, Variant, Params>(
        params, tmp_v, tmp_s, /*enable_pdl=*/false, stream);
  });

  if (err != cudaSuccess) {
    return InternalError("BatchPrefillWithPagedKVCacheDispatched failed: ",
                         cudaGetErrorString(err));
  }

  INFERX_CUDA_RETURN_IF_ERROR(cudaGetLastError());
  return OkStatus();
}

Status FlashInferPrefill::Prefill(
    const TensorView& q, const TensorView& k_cache, const TensorView& v_cache,
    const TensorView& qo_indptr, absl::Span<const int32_t> qo_indptr_host,
    const TensorView& kv_indices, const TensorView& kv_indptr,
    absl::Span<const int32_t> kv_indptr_host, const TensorView& last_page_len,
    const TensorView& out, float scale, cudaStream_t stream) {
  if (qo_indptr_host.empty()) {
    return InvalidArgumentError("qo_indptr is empty");
  }

  INFERX_RETURN_IF_ERROR(Plan(static_cast<int64_t>(qo_indptr_host.size()) - 1,
                              q.Dim(1), k_cache.Dim(2), q.Dim(2),
                              k_cache.Dim(1), qo_indptr_host, kv_indptr_host,
                              qo_indptr_host.back(), stream));

  return Run(q, k_cache, v_cache, qo_indptr, kv_indices, kv_indptr,
             last_page_len, out, scale, stream);
}

Status FlashInferPrefill::PrefillFp8(
    const TensorView& q, const TensorView& k_cache, const TensorView& v_cache,
    const TensorView& qo_indptr, absl::Span<const int32_t> qo_indptr_host,
    const TensorView& kv_indices, const TensorView& kv_indptr,
    absl::Span<const int32_t> kv_indptr_host, const TensorView& last_page_len,
    const TensorView& out, float scale, float k_scale, float v_scale,
    cudaStream_t stream) {
  if (qo_indptr_host.empty()) {
    return InvalidArgumentError("qo_indptr is empty");
  }

  INFERX_RETURN_IF_ERROR(CheckBf16(q, 3, "q"));
  INFERX_RETURN_IF_ERROR(CheckFp8(k_cache, 4, "k_cache"));
  INFERX_RETURN_IF_ERROR(CheckFp8(v_cache, 4, "v_cache"));
  INFERX_RETURN_IF_ERROR(CheckBf16(out, 3, "out"));
  INFERX_RETURN_IF_ERROR(CheckI32(qo_indptr, "qo_indptr"));
  INFERX_RETURN_IF_ERROR(CheckI32(kv_indices, "kv_indices"));
  INFERX_RETURN_IF_ERROR(CheckI32(kv_indptr, "kv_indptr"));
  INFERX_RETURN_IF_ERROR(CheckI32(last_page_len, "last_page_len"));

  const int64_t batch = static_cast<int64_t>(qo_indptr_host.size()) - 1;
  const int64_t q_heads = q.Dim(1);
  const int64_t head_dim = q.Dim(2);
  const int64_t kv_heads = k_cache.Dim(2);
  const int64_t page_size = k_cache.Dim(1);
  const int64_t total_rows = qo_indptr_host.back();

  std::vector<IdType> qo(qo_indptr_host.begin(), qo_indptr_host.end());
  std::vector<IdType> kv(kv_indptr_host.begin(), kv_indptr_host.end());

  // PrefillPlan is keyed on the *output* element size (sizeof_dtype_o), not the
  // KV dtype -- so the fp8-KV plan is identical to the bf16 one (Q and O stay
  // bf16). The KV dtype only matters to the dispatch below, which reads the
  // cache through Fp8 pointers.
  cudaError_t err = flashinfer::PrefillPlan<IdType>(
      impl_->float_workspace.data(), impl_->float_workspace.size(),
      impl_->int_workspace.data(), impl_->page_locked,
      impl_->int_workspace.size(), impl_->plan, qo.data(), kv.data(),
      static_cast<uint32_t>(total_rows), static_cast<uint32_t>(batch),
      static_cast<uint32_t>(q_heads), static_cast<uint32_t>(kv_heads),
      static_cast<uint32_t>(head_dim), static_cast<uint32_t>(head_dim),
      static_cast<uint32_t>(page_size),
      /*enable_cuda_graph=*/false, sizeof(bf16), /*window_left=*/-1,
      /*fixed_split_size=*/-1, /*disable_split_kv=*/false,
      /*num_colocated_ctas=*/0, stream);
  if (err != cudaSuccess) {
    return InternalError("PrefillPlan failed: ", cudaGetErrorString(err));
  }

  impl_->planned = true;
  impl_->planned_batch = batch;
  impl_->planned_q_heads = q_heads;
  impl_->planned_kv_heads = kv_heads;
  impl_->planned_head_dim = head_dim;
  impl_->planned_page_size = page_size;
  impl_->planned_total_rows = total_rows;

  if (out.Dim(0) != total_rows || out.Dim(1) != q_heads ||
      out.Dim(2) != head_dim) {
    return InvalidArgumentError("out is [", out.Dim(0), ",", out.Dim(1), ",",
                                out.Dim(2), "], expected [", total_rows, ",",
                                q_heads, ",", head_dim, "]");
  }
  if (k_cache.Dim(3) != head_dim) {
    return InvalidArgumentError("q head_dim ", head_dim,
                                " does not match the cache's ", k_cache.Dim(3));
  }

  flashinfer::paged_kv_t<Fp8, IdType> paged_kv(
      static_cast<uint32_t>(kv_heads), static_cast<uint32_t>(page_size),
      static_cast<uint32_t>(head_dim), static_cast<uint32_t>(batch),
      flashinfer::QKVLayout::kNHD, static_cast<Fp8*>(k_cache.Data()),
      static_cast<Fp8*>(v_cache.Data()),
      static_cast<IdType*>(kv_indices.Data()),
      static_cast<IdType*>(kv_indptr.Data()),
      static_cast<IdType*>(last_page_len.Data()));

  // Fold K's dequant scale into the query rather than the softmax scale. The
  // two are algebraically equivalent -- (q*k_scale)·k_fp8 == q·(k_scale*k_fp8)
  // == q·k -- but only the Q form keeps sm_scale at 1/sqrt(head_dim), which fa2
  // prefill's causal/length mask depends on: it suppresses masked K positions
  // inside exp2(logit*sm_scale_log2), and folding k_scale into sm_scale instead
  // drives sm_scale_log2 to ~1e-4 and lets those positions leak into the softmax
  // denominator (the bf16 path collapses the same way at an equally tiny
  // sm_scale, so this is not fp8-specific). Decode needs none of this: it
  // excludes invalid slots structurally, with no soft mask to defeat.
  INFERX_ASSIGN_OR_RETURN(
      DeviceBuffer q_scaled,
      DeviceBuffer::Allocate(q.Numel() * sizeof(bf16), DeviceId::Cuda(0)));
  ScaleBf16CopyKernel<<<GridFor(q.Numel()), 256, 0, stream>>>(
      static_cast<const bf16*>(q.Data()),
      reinterpret_cast<bf16*>(q_scaled.data()), q.Numel(), k_scale);
  INFERX_CUDA_RETURN_IF_ERROR(cudaGetLastError());

  Fp8Params params;
  params.q = reinterpret_cast<bf16*>(q_scaled.data());
  params.paged_kv = paged_kv;
  params.q_indptr = static_cast<IdType*>(qo_indptr.Data());
  params.maybe_mask_indptr = nullptr;
  params.maybe_q_rope_offset = nullptr;
  params.o = static_cast<bf16*>(out.Data());
  params.lse = nullptr;
  params.maybe_alibi_slopes = nullptr;
  params.num_qo_heads = static_cast<uint32_t>(q_heads);
  params.group_size =
      flashinfer::uint_fastdiv(static_cast<uint32_t>(q_heads / kv_heads));
  params.q_stride_n = static_cast<IdType>(q_heads * head_dim);
  params.q_stride_h = static_cast<IdType>(head_dim);
  params.window_left = -1;
  params.logits_soft_cap = 0.f;
  params.sm_scale = scale;
  params.rope_rcp_scale = 1.f;
  params.rope_rcp_theta = 1.f;
  params.maybe_prefix_len_ptr = nullptr;
  params.token_pos_in_items_len = 0;

  auto* int_base = static_cast<std::byte*>(impl_->int_workspace.data());
  const auto at = [&](size_t offset) {
    return reinterpret_cast<IdType*>(int_base + offset);
  };
  params.request_indices = at(impl_->plan.request_indices_offset);
  params.qo_tile_indices = at(impl_->plan.qo_tile_indices_offset);
  params.kv_tile_indices = at(impl_->plan.kv_tile_indices_offset);
  params.merge_indptr = at(impl_->plan.merge_indptr_offset);
  params.o_indptr = at(impl_->plan.o_indptr_offset);
  params.kv_chunk_size_ptr = at(impl_->plan.kv_chunk_size_ptr_offset);
  params.max_total_num_rows = static_cast<uint32_t>(impl_->plan.total_num_rows);
  params.total_num_rows = nullptr;
  params.padded_batch_size =
      static_cast<uint32_t>(impl_->plan.padded_batch_size);
  params.partition_kv = impl_->plan.split_kv;
  params.block_valid_mask =
      impl_->plan.split_kv ? reinterpret_cast<bool*>(
                                 int_base + impl_->plan.block_valid_mask_offset)
                           : nullptr;

  auto* float_base = static_cast<std::byte*>(impl_->float_workspace.data());
  bf16* tmp_v = nullptr;
  float* tmp_s = nullptr;
  if (impl_->plan.split_kv) {
    tmp_v = reinterpret_cast<bf16*>(float_base + impl_->plan.v_offset);
    tmp_s = reinterpret_cast<float*>(float_base + impl_->plan.s_offset);
  }

  DISPATCH_CTA_TILE_Q(impl_->plan.cta_tile_q, CTA_TILE_Q, {
    err = flashinfer::BatchPrefillWithPagedKVCacheDispatched<
        CTA_TILE_Q, /*HEAD_DIM_QK=*/128, /*HEAD_DIM_VO=*/128,
        flashinfer::PosEncodingMode::kNone, /*USE_FP16_QK_REDUCTION=*/false,
        flashinfer::MaskMode::kCausal, Variant, Fp8Params>(
        params, tmp_v, tmp_s, /*enable_pdl=*/false, stream);
  });

  if (err != cudaSuccess) {
    return InternalError("BatchPrefillWithPagedKVCacheDispatched failed: ",
                         cudaGetErrorString(err));
  }

  INFERX_CUDA_RETURN_IF_ERROR(cudaGetLastError());

  // V's dequant scale, applied post-kernel (not fused, as in FlashInfer).
  if (v_scale != 1.0f) {
    ScaleBf16Kernel<<<GridFor(out.Numel()), 256, 0, stream>>>(
        static_cast<bf16*>(out.Data()), out.Numel(), v_scale);
    INFERX_CUDA_RETURN_IF_ERROR(cudaGetLastError());
  }

  return OkStatus();
}

}  // namespace inferx::kernels
