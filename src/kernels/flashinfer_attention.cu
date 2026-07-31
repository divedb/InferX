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

#include "inferx/core/cuda_utils.h"
#include "inferx/core/device_buffer.h"

namespace inferx::kernels {
namespace {

using bf16 = __nv_bfloat16;
using IdType = int32_t;

using Params = flashinfer::BatchDecodeParams<bf16, bf16, bf16, IdType>;

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

}  // namespace

struct FlashInferDecode::Impl {
  DeviceBuffer float_workspace;
  DeviceBuffer int_workspace;
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

Status FlashInferDecode::Decode(const TensorView& q, const TensorView& k_cache,
                                const TensorView& v_cache,
                                const TensorView& kv_indices,
                                const TensorView& kv_indptr,
                                absl::Span<const int32_t> kv_indptr_host,
                                const TensorView& last_page_len,
                                const TensorView& out, float scale,
                                cudaStream_t stream) {
  INFERX_RETURN_IF_ERROR(CheckBf16(q, 3, "q"));
  INFERX_RETURN_IF_ERROR(CheckBf16(k_cache, 4, "k_cache"));
  INFERX_RETURN_IF_ERROR(CheckBf16(v_cache, 4, "v_cache"));
  INFERX_RETURN_IF_ERROR(CheckBf16(out, 3, "out"));
  INFERX_RETURN_IF_ERROR(CheckI32(kv_indices, "kv_indices"));
  INFERX_RETURN_IF_ERROR(CheckI32(kv_indptr, "kv_indptr"));
  INFERX_RETURN_IF_ERROR(CheckI32(last_page_len, "last_page_len"));

  const int64_t batch = q.Dim(0);
  const int64_t q_heads = q.Dim(1);
  const int64_t head_dim = q.Dim(2);
  const int64_t page_size = k_cache.Dim(1);
  const int64_t kv_heads = k_cache.Dim(2);

  if (k_cache.Dim(3) != head_dim) {
    return InvalidArgumentError("q head_dim ", head_dim,
                                " does not match the cache's ", k_cache.Dim(3));
  }
  if (kv_heads == 0 || q_heads % kv_heads != 0) {
    return InvalidArgumentError("q_heads ", q_heads,
                                " is not a multiple of kv_heads ", kv_heads);
  }
  if (kv_indptr.Dim(0) != batch + 1) {
    return InvalidArgumentError("kv_indptr has ", kv_indptr.Dim(0),
                                " entries, expected batch + 1 = ", batch + 1);
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

  // Only head_dim 128 is instantiated. Every head dimension is a separate
  // template instantiation and therefore separate compile time and binary size;
  // 128 is what Qwen2.5 and Llama both use, so the others are not paid for
  // until something needs them.
  if (head_dim != 128) {
    return UnimplementedError(
        "FlashInfer decode is instantiated for head_dim 128 only, got ",
        head_dim, "; add an instantiation in flashinfer_attention.cu");
  }

  // Checked rather than trusted: the planner reads this array to decide how to
  // split the work, and a host copy that disagrees with the device one produces
  // a plan for a batch that is not the one being run -- wrong output, no error.
  if (static_cast<int64_t>(kv_indptr_host.size()) != batch + 1) {
    return InvalidArgumentError("kv_indptr_host has ", kv_indptr_host.size(),
                                " entries, expected batch + 1 = ", batch + 1);
  }

  if (batch == 0) return OkStatus();

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
  params.window_left = -1;      // no sliding window
  params.logits_soft_cap = 0.f;
  params.sm_scale = scale;
  // RoPE is applied before the cache write in our pipeline, so the kernel must
  // not apply it again. kNone below makes that explicit; these are unused.
  params.rope_rcp_scale = 1.f;
  params.rope_rcp_theta = 1.f;

  constexpr auto kPos = flashinfer::PosEncodingMode::kNone;

  flashinfer::DecodePlanInfo plan_info;

  // The work estimator is templated on the GQA group size, so it has to be
  // selected at runtime through FlashInfer's own dispatch macro -- which is a
  // chain of `if (group_size == N) { constexpr size_t GROUP_SIZE = N; ... }`,
  // instantiating the estimator once per supported ratio. Qwen2.5-3B is 16 q
  // heads over 2 kv heads, so it lands on 8.
  cudaError_t plan_status = cudaSuccess;
  bool group_size_supported = false;

  DISPATCH_GQA_GROUP_SIZE(q_heads / kv_heads, GROUP_SIZE, {
    group_size_supported = true;

    auto work_estimation =
        flashinfer::BatchDecodeWithPagedKVCacheWorkEstimationDispatched<
            GROUP_SIZE, 128, kPos, Variant, Params>;

    plan_status = flashinfer::DecodePlan<128, kPos, Variant, Params>(
        impl_->float_workspace.data(), impl_->float_workspace.size(),
        impl_->int_workspace.data(), impl_->page_locked,
        impl_->int_workspace.size(), plan_info,
        const_cast<IdType*>(kv_indptr_host.data()),
        static_cast<uint32_t>(batch), static_cast<uint32_t>(q_heads),
        static_cast<uint32_t>(page_size), /*enable_cuda_graph=*/false, stream,
        work_estimation);
  });

  if (!group_size_supported) {
    return UnimplementedError("FlashInfer has no instantiation for a GQA group "
                              "size of ", q_heads / kv_heads);
  }

  INFERX_CUDA_RETURN_IF_ERROR(plan_status);

  // The plan hands back byte offsets into the int workspace rather than
  // pointers, so the caller can keep one allocation and re-plan into it.
  auto* int_base = static_cast<std::byte*>(impl_->int_workspace.data());
  const auto at = [&](size_t offset) {
    return reinterpret_cast<IdType*>(int_base + offset);
  };

  params.request_indices = at(plan_info.request_indices_offset);
  params.kv_tile_indices = at(plan_info.kv_tile_indices_offset);
  params.o_indptr = at(plan_info.o_indptr_offset);
  params.kv_chunk_size_ptr = at(plan_info.kv_chunk_size_ptr_offset);
  params.padded_batch_size = static_cast<uint32_t>(plan_info.padded_batch_size);
  params.partition_kv = plan_info.split_kv;

  params.block_valid_mask =
      plan_info.split_kv
          ? reinterpret_cast<bool*>(int_base + plan_info.block_valid_mask_offset)
          : nullptr;

  auto* float_base = static_cast<std::byte*>(impl_->float_workspace.data());

  bf16* tmp_v = nullptr;
  float* tmp_s = nullptr;
  if (plan_info.split_kv) {
    tmp_v = reinterpret_cast<bf16*>(float_base + plan_info.v_offset);
    tmp_s = reinterpret_cast<float*>(float_base + plan_info.s_offset);
  }

  INFERX_CUDA_RETURN_IF_ERROR(
      (flashinfer::BatchDecodeWithPagedKVCacheDispatched<128, kPos, Variant,
                                                         Params>(
          params, tmp_v, tmp_s, /*enable_pdl=*/false, stream)));

  return OkStatus();
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
