#include "inferx/kernels/flashinfer_attention.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <utility>

// FlashInfer's template layer. These are the only headers we take from it; the
// wrapper layer under flashinfer/ (python, JIT, torch bindings) is deliberately
// not on the include path.
#include <flashinfer/attention/decode.cuh>
#include <flashinfer/attention/default_decode_params.cuh>
#include <flashinfer/attention/scheduler.cuh>
#include <flashinfer/attention/variants.cuh>
#include <flashinfer/page.cuh>
#include <flashinfer/pos_enc.cuh>
#include <flashinfer/utils.cuh>

#include "inferx/backends/cuda/cuda_utils.h"
#include "inferx/core/device_buffer.h"

namespace inferx::kernels {
namespace {

using bf16 = __nv_bfloat16;
using IdType = int32_t;

using Params = flashinfer::BatchDecodeParams<bf16, bf16, bf16, IdType>;

// FP8-e4m3 KV: Q and O stay bf16, K/V are fp8 in the paged cache, accumulation
// is fp32 inside the kernel. The fa2 decode upcasts fp8 on load, so this needs
// no fp8 tensor cores and is sm_89-capable.
using Fp8 = __nv_fp8_e4m3;
using Fp8Params = flashinfer::BatchDecodeParams<bf16, Fp8, bf16, IdType>;

// No custom mask, no sliding window, no soft cap, no ALiBi. Decode is causal by
// construction here -- a sequence's pages hold exactly its own history -- so
// there is no mask to apply.
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

// The fp8 KV cache check. Same shape rules as the bf16 path; only the element
// type differs.
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

// Applies the V dequant scale to the output. FlashInfer folds K's scale into
// the softmax but leaves V's scale out of the kernel entirely, so it has to be
// applied here. bf16 in place; one pass.
__global__ void ScaleBf16Kernel(bf16* x, int64_t n, float scale) {
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < n; i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    x[i] = __float2bfloat16(__bfloat162float(x[i]) * scale);
  }
}

int GridFor(int64_t n) {
  constexpr int kBlock = 256;
  const int64_t want = (n + kBlock - 1) / kBlock;
  return static_cast<int>(want < 1024 ? (want < 1 ? 1 : want) : 1024);
}

}  // namespace

struct FlashInferDecode::Impl {
  DeviceBuffer float_workspace;
  DeviceBuffer int_workspace;

  // The last plan, and the shape it was built for. Run() checks against these
  // rather than trusting the caller: a kernel launched with a plan for another
  // batch reads past the end of its own indices.
  flashinfer::DecodePlanInfo plan;
  bool planned = false;
  // The plan's workspace split is dtype-dependent (fp8 vectors 16-wide against
  // bf16's 8), so a bf16 plan is not valid for an fp8 run or vice versa. This
  // tracks which was planned.
  bool planned_fp8 = false;
  int64_t planned_batch = 0;
  int64_t planned_q_heads = 0;
  int64_t planned_kv_heads = 0;
  int64_t planned_head_dim = 0;
  int64_t planned_page_size = 0;
  // DecodePlan writes its integer plan through a page-locked staging buffer
  // before the async copy to device, so this cannot be ordinary host memory.
  void* page_locked = nullptr;
  size_t page_locked_bytes = 0;

  ~Impl() {
    if (page_locked != nullptr) cudaFreeHost(page_locked);
  }
};

StatusOr<FlashInferDecode> FlashInferDecode::Create(
    size_t float_workspace_bytes, size_t int_workspace_bytes) {
  auto impl = std::make_unique<Impl>();

  INFERX_ASSIGN_OR_RETURN(
      impl->float_workspace,
      DeviceBuffer::Allocate(float_workspace_bytes, DeviceId::Cuda(0)));
  INFERX_ASSIGN_OR_RETURN(
      impl->int_workspace,
      DeviceBuffer::Allocate(int_workspace_bytes, DeviceId::Cuda(0)));

  impl->page_locked_bytes = int_workspace_bytes;
  INFERX_CUDA_RETURN_IF_ERROR(
      cudaHostAlloc(&impl->page_locked, int_workspace_bytes,
                    cudaHostAllocDefault));

  return FlashInferDecode(std::move(impl));
}

FlashInferDecode::FlashInferDecode(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}
FlashInferDecode::~FlashInferDecode() = default;
FlashInferDecode::FlashInferDecode(FlashInferDecode&&) noexcept = default;
FlashInferDecode& FlashInferDecode::operator=(FlashInferDecode&&) noexcept =
    default;

Status FlashInferDecode::Plan(int64_t batch, int64_t q_heads,
                              int64_t kv_heads, int64_t head_dim,
                              int64_t page_size,
                              absl::Span<const int32_t> kv_indptr_host,
                              bool graph_safe, Stream stream) {
  if (batch <= 0 || q_heads <= 0 || kv_heads <= 0 || page_size <= 0) {
    return InvalidArgumentError("degenerate plan shape: batch=", batch,
                                " q_heads=", q_heads, " kv_heads=", kv_heads,
                                " page_size=", page_size);
  }

  if (q_heads % kv_heads != 0) {
    return InvalidArgumentError("q_heads ", q_heads,
                                " is not a multiple of kv_heads ", kv_heads);
  }

  if (head_dim != 128) {
    return UnimplementedError(
        "FlashInfer decode is instantiated for head_dim 128 only, got ",
        head_dim, "; add an instantiation in flashinfer_attention.cu");
  }

  if (static_cast<int64_t>(kv_indptr_host.size()) != batch + 1) {
    return InvalidArgumentError("kv_indptr_host has ", kv_indptr_host.size(),
                                " entries, expected batch + 1 = ", batch + 1);
  }

  constexpr auto kPos = flashinfer::PosEncodingMode::kNone;

  cudaError_t status = cudaSuccess;
  bool group_size_supported = false;

  DISPATCH_GQA_GROUP_SIZE(q_heads / kv_heads, GROUP_SIZE, {
    group_size_supported = true;

    auto work_estimation =
        flashinfer::BatchDecodeWithPagedKVCacheWorkEstimationDispatched<
            GROUP_SIZE, 128, kPos, Variant, Params>;

    status = flashinfer::DecodePlan<128, kPos, Variant, Params>(
        impl_->float_workspace.data(), impl_->float_workspace.size(),
        impl_->int_workspace.data(), impl_->page_locked,
        impl_->int_workspace.size(), impl_->plan,
        const_cast<IdType*>(kv_indptr_host.data()),
        static_cast<uint32_t>(batch), static_cast<uint32_t>(q_heads),
        static_cast<uint32_t>(page_size), graph_safe, stream,
        work_estimation);
  });

  if (!group_size_supported) {
    return UnimplementedError("FlashInfer has no instantiation for a GQA group "
                              "size of ", q_heads / kv_heads);
  }

  INFERX_CUDA_RETURN_IF_ERROR(status);

  impl_->planned = true;
  impl_->planned_fp8 = false;
  impl_->planned_batch = batch;
  impl_->planned_q_heads = q_heads;
  impl_->planned_kv_heads = kv_heads;
  impl_->planned_head_dim = head_dim;
  impl_->planned_page_size = page_size;

  return OkStatus();
}

Status FlashInferDecode::Run(const TensorView& q, const TensorView& k_cache,
                             const TensorView& v_cache,
                             const TensorView& kv_indices,
                             const TensorView& kv_indptr,
                             const TensorView& last_page_len,
                             const TensorView& out, float scale,
                             Stream stream) {
  INFERX_RETURN_IF_ERROR(CheckBf16(q, 3, "q"));
  INFERX_RETURN_IF_ERROR(CheckBf16(k_cache, 4, "k_cache"));
  INFERX_RETURN_IF_ERROR(CheckBf16(v_cache, 4, "v_cache"));
  INFERX_RETURN_IF_ERROR(CheckBf16(out, 3, "out"));
  INFERX_RETURN_IF_ERROR(CheckI32(kv_indices, "kv_indices"));
  INFERX_RETURN_IF_ERROR(CheckI32(kv_indptr, "kv_indptr"));
  INFERX_RETURN_IF_ERROR(CheckI32(last_page_len, "last_page_len"));

  if (!impl_->planned) {
    return FailedPreconditionError("Run() called before Plan()");
  }
  if (impl_->planned_fp8) {
    return FailedPreconditionError(
        "Run() serves the bf16 path but the last plan was FP8 (PlanFp8); "
        "call Plan or RunFp8");
  }

  const int64_t batch = q.Dim(0);
  const int64_t q_heads = q.Dim(1);
  const int64_t head_dim = q.Dim(2);
  const int64_t page_size = k_cache.Dim(1);
  const int64_t kv_heads = k_cache.Dim(2);

  if (batch != impl_->planned_batch || q_heads != impl_->planned_q_heads ||
      kv_heads != impl_->planned_kv_heads ||
      head_dim != impl_->planned_head_dim ||
      page_size != impl_->planned_page_size) {
    return InvalidArgumentError(
        "these tensors are [batch=", batch, " q_heads=", q_heads,
        " kv_heads=", kv_heads, " head_dim=", head_dim,
        " page=", page_size, "] but the last plan was for [batch=",
        impl_->planned_batch, " q_heads=", impl_->planned_q_heads,
        " kv_heads=", impl_->planned_kv_heads, " head_dim=",
        impl_->planned_head_dim, " page=", impl_->planned_page_size, "]");
  }

  if (k_cache.Dim(3) != head_dim) {
    return InvalidArgumentError("q head_dim ", head_dim,
                                " does not match the cache's ", k_cache.Dim(3));
  }
  if (kv_indptr.Dim(0) != batch + 1) {
    return InvalidArgumentError("kv_indptr has ", kv_indptr.Dim(0),
                                " entries, expected ", batch + 1);
  }
  if (last_page_len.Dim(0) != batch) {
    return InvalidArgumentError("last_page_len has ", last_page_len.Dim(0),
                                " entries, expected ", batch);
  }
  if (out.Dim(0) != batch || out.Dim(1) != q_heads ||
      out.Dim(2) != head_dim) {
    return InvalidArgumentError("out is ", out.GetShape().ToString(),
                                ", expected [", batch, ", ", q_heads, ", ",
                                head_dim, "]");
  }

  flashinfer::paged_kv_t<bf16, IdType> paged_kv(
      static_cast<uint32_t>(kv_heads), static_cast<uint32_t>(page_size),
      static_cast<uint32_t>(head_dim), static_cast<uint32_t>(batch),
      flashinfer::QKVLayout::kNHD, static_cast<bf16*>(k_cache.Data()),
      static_cast<bf16*>(v_cache.Data()),
      static_cast<IdType*>(kv_indices.Data()),
      static_cast<IdType*>(kv_indptr.Data()),
      static_cast<IdType*>(last_page_len.Data()));

  Params params;
  params.q = static_cast<bf16*>(q.Data());
  params.paged_kv = paged_kv;
  params.o = static_cast<bf16*>(out.Data());
  params.lse = nullptr;
  params.num_qo_heads = static_cast<uint32_t>(q_heads);
  params.q_stride_n = static_cast<IdType>(q_heads * head_dim);
  params.q_stride_h = static_cast<IdType>(head_dim);
  params.window_left = -1;
  params.logits_soft_cap = 0.f;
  params.sm_scale = scale;
  params.rope_rcp_scale = 1.f;
  params.rope_rcp_theta = 1.f;

  // Offsets rather than pointers is what makes this replayable: the plan hands
  // back positions in a workspace this object owns, so the addresses below are
  // the same on every step and only their contents change.
  auto* int_base = static_cast<std::byte*>(impl_->int_workspace.data());
  const auto at = [&](size_t offset) {
    return reinterpret_cast<IdType*>(int_base + offset);
  };

  params.request_indices = at(impl_->plan.request_indices_offset);
  params.kv_tile_indices = at(impl_->plan.kv_tile_indices_offset);
  params.o_indptr = at(impl_->plan.o_indptr_offset);
  params.kv_chunk_size_ptr = at(impl_->plan.kv_chunk_size_ptr_offset);
  params.padded_batch_size =
      static_cast<uint32_t>(impl_->plan.padded_batch_size);
  params.partition_kv = impl_->plan.split_kv;

  params.block_valid_mask =
      impl_->plan.split_kv
          ? reinterpret_cast<bool*>(int_base +
                                    impl_->plan.block_valid_mask_offset)
          : nullptr;

  auto* float_base = static_cast<std::byte*>(impl_->float_workspace.data());

  bf16* tmp_v = nullptr;
  float* tmp_s = nullptr;
  if (impl_->plan.split_kv) {
    tmp_v = reinterpret_cast<bf16*>(float_base + impl_->plan.v_offset);
    tmp_s = reinterpret_cast<float*>(float_base + impl_->plan.s_offset);
  }

  constexpr auto kPos = flashinfer::PosEncodingMode::kNone;

  INFERX_CUDA_RETURN_IF_ERROR(
      (flashinfer::BatchDecodeWithPagedKVCacheDispatched<128, kPos, Variant,
                                                         Params>(
          params, tmp_v, tmp_s, /*enable_pdl=*/false, stream)));

  return OkStatus();
}

Status FlashInferDecode::Decode(const TensorView& q, const TensorView& k_cache,
                                const TensorView& v_cache,
                                const TensorView& kv_indices,
                                const TensorView& kv_indptr,
                                absl::Span<const int32_t> kv_indptr_host,
                                const TensorView& last_page_len,
                                const TensorView& out, float scale,
                                Stream stream) {
  INFERX_RETURN_IF_ERROR(CheckBf16(q, 3, "q"));
  INFERX_RETURN_IF_ERROR(CheckBf16(k_cache, 4, "k_cache"));

  const int64_t batch = q.Dim(0);
  if (batch == 0) return OkStatus();

  // Not graph-safe: the one-shot form is for callers that are not capturing,
  // and the fixed-shape mode costs a reduction pass they need not pay.
  INFERX_RETURN_IF_ERROR(Plan(batch, q.Dim(1), k_cache.Dim(2), q.Dim(2),
                              k_cache.Dim(1), kv_indptr_host,
                              /*graph_safe=*/false, stream));

  return Run(q, k_cache, v_cache, kv_indices, kv_indptr, last_page_len, out,
             scale, stream);
}

Status FlashInferDecode::PlanFp8(int64_t batch, int64_t q_heads,
                                 int64_t kv_heads, int64_t head_dim,
                                 int64_t page_size,
                                 absl::Span<const int32_t> kv_indptr_host,
                                 bool graph_safe, Stream stream) {
  if (batch <= 0 || q_heads <= 0 || kv_heads <= 0 || page_size <= 0) {
    return InvalidArgumentError("degenerate plan shape: batch=", batch,
                                " q_heads=", q_heads, " kv_heads=", kv_heads,
                                " page_size=", page_size);
  }

  if (q_heads % kv_heads != 0) {
    return InvalidArgumentError("q_heads ", q_heads,
                                " is not a multiple of kv_heads ", kv_heads);
  }

  if (head_dim != 128) {
    return UnimplementedError(
        "FlashInfer decode is instantiated for head_dim 128 only, got ",
        head_dim, "; add an instantiation in flashinfer_attention.cu");
  }

  if (static_cast<int64_t>(kv_indptr_host.size()) != batch + 1) {
    return InvalidArgumentError("kv_indptr_host has ", kv_indptr_host.size(),
                                " entries, expected batch + 1 = ", batch + 1);
  }

  constexpr auto kPos = flashinfer::PosEncodingMode::kNone;

  cudaError_t status = cudaSuccess;
  bool group_size_supported = false;

  DISPATCH_GQA_GROUP_SIZE(q_heads / kv_heads, GROUP_SIZE, {
    group_size_supported = true;

    auto work_estimation =
        flashinfer::BatchDecodeWithPagedKVCacheWorkEstimationDispatched<
            GROUP_SIZE, 128, kPos, Variant, Fp8Params>;

    status = flashinfer::DecodePlan<128, kPos, Variant, Fp8Params>(
        impl_->float_workspace.data(), impl_->float_workspace.size(),
        impl_->int_workspace.data(), impl_->page_locked,
        impl_->int_workspace.size(), impl_->plan,
        const_cast<IdType*>(kv_indptr_host.data()),
        static_cast<uint32_t>(batch), static_cast<uint32_t>(q_heads),
        static_cast<uint32_t>(page_size), graph_safe, stream, work_estimation);
  });

  if (!group_size_supported) {
    return UnimplementedError("FlashInfer has no instantiation for a GQA group "
                              "size of ",
                              q_heads / kv_heads);
  }
  INFERX_CUDA_RETURN_IF_ERROR(status);

  impl_->planned = true;
  impl_->planned_fp8 = true;
  impl_->planned_batch = batch;
  impl_->planned_q_heads = q_heads;
  impl_->planned_kv_heads = kv_heads;
  impl_->planned_head_dim = head_dim;
  impl_->planned_page_size = page_size;
  return OkStatus();
}

Status FlashInferDecode::RunFp8(const TensorView& q, const TensorView& k_cache,
                                const TensorView& v_cache,
                                const TensorView& kv_indices,
                                const TensorView& kv_indptr,
                                const TensorView& last_page_len,
                                const TensorView& out, float scale,
                                float k_scale, float v_scale,
                                Stream stream) {
  INFERX_RETURN_IF_ERROR(CheckBf16(q, 3, "q"));
  INFERX_RETURN_IF_ERROR(CheckFp8(k_cache, 4, "k_cache"));
  INFERX_RETURN_IF_ERROR(CheckFp8(v_cache, 4, "v_cache"));
  INFERX_RETURN_IF_ERROR(CheckBf16(out, 3, "out"));
  INFERX_RETURN_IF_ERROR(CheckI32(kv_indices, "kv_indices"));
  INFERX_RETURN_IF_ERROR(CheckI32(kv_indptr, "kv_indptr"));
  INFERX_RETURN_IF_ERROR(CheckI32(last_page_len, "last_page_len"));

  if (!impl_->planned) {
    return FailedPreconditionError("RunFp8() called before PlanFp8()");
  }
  if (!impl_->planned_fp8) {
    return FailedPreconditionError(
        "RunFp8() serves the FP8 path but the last plan was bf16 (Plan); "
        "call PlanFp8");
  }

  const int64_t batch = q.Dim(0);
  const int64_t q_heads = q.Dim(1);
  const int64_t head_dim = q.Dim(2);
  const int64_t page_size = k_cache.Dim(1);
  const int64_t kv_heads = k_cache.Dim(2);

  if (batch != impl_->planned_batch || q_heads != impl_->planned_q_heads ||
      kv_heads != impl_->planned_kv_heads ||
      head_dim != impl_->planned_head_dim ||
      page_size != impl_->planned_page_size) {
    return InvalidArgumentError(
        "FP8 tensors are [batch=", batch, " q_heads=", q_heads,
        " kv_heads=", kv_heads, " head_dim=", head_dim,
        " page=", page_size, "] but the last FP8 plan was for [batch=",
        impl_->planned_batch, " q_heads=", impl_->planned_q_heads,
        " kv_heads=", impl_->planned_kv_heads, " head_dim=",
        impl_->planned_head_dim, " page=", impl_->planned_page_size, "]");
  }
  if (k_cache.Dim(3) != head_dim) {
    return InvalidArgumentError("q head_dim ", head_dim,
                                " does not match the cache's ", k_cache.Dim(3));
  }
  if (kv_indptr.Dim(0) != batch + 1) {
    return InvalidArgumentError("kv_indptr has ", kv_indptr.Dim(0),
                                " entries, expected ", batch + 1);
  }
  if (last_page_len.Dim(0) != batch) {
    return InvalidArgumentError("last_page_len has ", last_page_len.Dim(0),
                                " entries, expected ", batch);
  }
  if (out.Dim(0) != batch || out.Dim(1) != q_heads ||
      out.Dim(2) != head_dim) {
    return InvalidArgumentError("out is ", out.GetShape().ToString(),
                                ", expected [", batch, ", ", q_heads, ", ",
                                head_dim, "]");
  }

  flashinfer::paged_kv_t<Fp8, IdType> paged_kv(
      static_cast<uint32_t>(kv_heads), static_cast<uint32_t>(page_size),
      static_cast<uint32_t>(head_dim), static_cast<uint32_t>(batch),
      flashinfer::QKVLayout::kNHD, static_cast<Fp8*>(k_cache.Data()),
      static_cast<Fp8*>(v_cache.Data()),
      static_cast<IdType*>(kv_indices.Data()),
      static_cast<IdType*>(kv_indptr.Data()),
      static_cast<IdType*>(last_page_len.Data()));

  Fp8Params params;
  params.q = static_cast<bf16*>(q.Data());
  params.paged_kv = paged_kv;
  params.o = static_cast<bf16*>(out.Data());
  params.lse = nullptr;
  params.num_qo_heads = static_cast<uint32_t>(q_heads);
  params.q_stride_n = static_cast<IdType>(q_heads * head_dim);
  params.q_stride_h = static_cast<IdType>(head_dim);
  params.window_left = -1;
  params.logits_soft_cap = 0.f;
  // K's dequant scale is absorbed into the softmax scale: attention is invariant
  // to a rescaling of K, so sm_scale * k_scale lands the QK products in the
  // right units before the softmax normalizes them away.
  params.sm_scale = scale * k_scale;
  params.rope_rcp_scale = 1.f;
  params.rope_rcp_theta = 1.f;

  auto* int_base = static_cast<std::byte*>(impl_->int_workspace.data());
  const auto at = [&](size_t offset) {
    return reinterpret_cast<IdType*>(int_base + offset);
  };

  params.request_indices = at(impl_->plan.request_indices_offset);
  params.kv_tile_indices = at(impl_->plan.kv_tile_indices_offset);
  params.o_indptr = at(impl_->plan.o_indptr_offset);
  params.kv_chunk_size_ptr = at(impl_->plan.kv_chunk_size_ptr_offset);
  params.padded_batch_size =
      static_cast<uint32_t>(impl_->plan.padded_batch_size);
  params.partition_kv = impl_->plan.split_kv;

  params.block_valid_mask =
      impl_->plan.split_kv
          ? reinterpret_cast<bool*>(int_base +
                                    impl_->plan.block_valid_mask_offset)
          : nullptr;

  auto* float_base = static_cast<std::byte*>(impl_->float_workspace.data());

  bf16* tmp_v = nullptr;
  float* tmp_s = nullptr;
  if (impl_->plan.split_kv) {
    tmp_v = reinterpret_cast<bf16*>(float_base + impl_->plan.v_offset);
    tmp_s = reinterpret_cast<float*>(float_base + impl_->plan.s_offset);
  }

  constexpr auto kPos = flashinfer::PosEncodingMode::kNone;

  INFERX_CUDA_RETURN_IF_ERROR(
      (flashinfer::BatchDecodeWithPagedKVCacheDispatched<128, kPos, Variant,
                                                         Fp8Params>(
          params, tmp_v, tmp_s, /*enable_pdl=*/false, stream)));

  // V's dequant scale is not in the kernel -- apply it to the output here.
  // Skipped at 1.0 both because that is the common fp8-with-unit-scale case and
  // to avoid a launch that does nothing.
  if (v_scale != 1.0f) {
    ScaleBf16Kernel<<<GridFor(out.Numel()), 256, 0, stream>>>(
        static_cast<bf16*>(out.Data()), out.Numel(), v_scale);
    INFERX_CUDA_RETURN_IF_ERROR(cudaGetLastError());
  }

  return OkStatus();
}

Status FlashInferDecode::DecodeFp8(
    const TensorView& q, const TensorView& k_cache, const TensorView& v_cache,
    const TensorView& kv_indices, const TensorView& kv_indptr,
    absl::Span<const int32_t> kv_indptr_host, const TensorView& last_page_len,
    const TensorView& out, float scale, float k_scale, float v_scale,
    Stream stream) {
  INFERX_RETURN_IF_ERROR(CheckBf16(q, 3, "q"));
  INFERX_RETURN_IF_ERROR(CheckFp8(k_cache, 4, "k_cache"));

  const int64_t batch = q.Dim(0);
  if (batch == 0) return OkStatus();

  // Not graph-safe: the one-shot form mirrors Decode() -- plan in fixed-shape
  // mode off, launch in the same call. Graph callers use PlanFp8 + RunFp8.
  INFERX_RETURN_IF_ERROR(PlanFp8(batch, q.Dim(1), k_cache.Dim(2), q.Dim(2),
                                 k_cache.Dim(1), kv_indptr_host,
                                 /*graph_safe=*/false, stream));

  return RunFp8(q, k_cache, v_cache, kv_indices, kv_indptr, last_page_len, out,
                scale, k_scale, v_scale, stream);
}

Status BuildCsrBlockTable(const std::vector<int32_t>& block_table,
                          int64_t num_seqs, int64_t max_blocks_per_seq,
                          const std::vector<int32_t>& blocks_used,
                          std::vector<int32_t>* out_indices,
                          std::vector<int32_t>* out_indptr) {
  if (static_cast<int64_t>(block_table.size()) !=
      num_seqs * max_blocks_per_seq) {
    return InvalidArgumentError("block_table has ", block_table.size(),
                                " entries, expected ",
                                num_seqs * max_blocks_per_seq);
  }

  if (static_cast<int64_t>(blocks_used.size()) != num_seqs) {
    return InvalidArgumentError("blocks_used has ", blocks_used.size(),
                                " entries, expected ", num_seqs);
  }

  out_indices->clear();
  out_indptr->clear();
  out_indptr->push_back(0);

  for (int64_t s = 0; s < num_seqs; ++s) {
    const int32_t used = blocks_used[static_cast<size_t>(s)];

    if (used < 0 || used > max_blocks_per_seq) {
      return InvalidArgumentError("sequence ", s, " claims ", used,
                                  " blocks, outside [0, ", max_blocks_per_seq,
                                  "]");
    }

    for (int32_t b = 0; b < used; ++b) {
      out_indices->push_back(
          block_table[static_cast<size_t>(s * max_blocks_per_seq + b)]);
    }

    out_indptr->push_back(static_cast<int32_t>(out_indices->size()));
  }

  return OkStatus();
}

}  // namespace inferx::kernels
