// Paged decode attention: FlashInfer against our reference kernel.
//
// M3 integrated FlashInfer on the assumption it would be faster, and left that
// unmeasured. This measures it. The two are numerically checked against each
// other in tests/kernel/flashinfer_test.cc; what is compared here is only cost.
//
// The sweep is the decode matrix: batch size against context length, at
// Qwen2.5-3B's geometry. Decode is where an inference engine spends its life,
// and it is the shape our kernel is worst at by construction -- one block per
// (query, head) walking every key serially, with no tiling and no split-KV. The
// question is how much that costs, and where.
//
// Both kernels read the identical cache, so no data movement is in the number.
// Prefill is not compared: only decode is integrated so far.

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "bench/cuda_timer.h"
#include "inferx/backends/cuda/cuda_utils.h"
#include "inferx/core/device_buffer.h"
#include "inferx/core/shape.h"
#include "inferx/core/status.h"
#include "inferx/core/tensor_view.h"
#include "absl/types/span.h"
#include "inferx/ops/flashinfer_attention.h"
#include "inferx/ops/layers.h"

namespace inferx::bench {
namespace {

using bf16 = __nv_bfloat16;

// Qwen2.5-3B: 16 query heads over 2 KV heads, head_dim 128, 16-token pages.
constexpr int64_t kQHeads = 16;
constexpr int64_t kKvHeads = 2;
constexpr int64_t kHeadDim = 128;
constexpr int64_t kPageSize = 16;

constexpr int64_t kBatches[] = {1, 8, 32, 128, 256};
constexpr int64_t kContexts[] = {128, 512, 2048, 8192};

/// Keeps device allocations alive for one configuration.
struct Arena {
  std::vector<DeviceBuffer> bufs;

  StatusOr<TensorView> Alloc(DataType dtype, const Shape& shape) {
    INFERX_ASSIGN_OR_RETURN(
        DeviceBuffer buf,
        DeviceBuffer::Allocate(
            static_cast<size_t>(DataTypeByteSize(dtype, shape.Numel())),
            DeviceId::Cuda(0)));

    INFERX_CUDA_RETURN_IF_ERROR(cudaMemset(buf.data(), 0, buf.size()));
    bufs.push_back(std::move(buf));

    return TensorView::Create(bufs.back().data(), dtype, shape,
                              DeviceId::Cuda(0));
  }

  StatusOr<TensorView> Upload(const std::vector<int32_t>& host) {
    INFERX_ASSIGN_OR_RETURN(
        DeviceBuffer buf,
        DeviceBuffer::Allocate(host.size() * sizeof(int32_t),
                               DeviceId::Cuda(0)));

    INFERX_CUDA_RETURN_IF_ERROR(cudaMemcpy(buf.data(), host.data(),
                                           host.size() * sizeof(int32_t),
                                           cudaMemcpyHostToDevice));
    bufs.push_back(std::move(buf));

    return TensorView::Create(
        bufs.back().data(), DataType::kInt32,
        Shape({static_cast<int64_t>(host.size())}), DeviceId::Cuda(0));
  }
};

Status RunOne(ops::FlashInferDecode* fi, int64_t batch, int64_t context,
              int warmup, int iters, bool* ours_ran) {
  Arena arena;

  const int64_t blocks_per_seq = (context + kPageSize - 1) / kPageSize;
  const int64_t num_blocks = batch * blocks_per_seq;
  const float scale = 1.0f / std::sqrt(static_cast<float>(kHeadDim));

  // Every sequence the same length: a ragged batch would measure the tail
  // rather than the shape, and the point here is the shape.
  INFERX_ASSIGN_OR_RETURN(
      const TensorView k_cache,
      arena.Alloc(DataType::kBFloat16,
                  Shape({num_blocks, kPageSize, kKvHeads, kHeadDim})));
  INFERX_ASSIGN_OR_RETURN(
      const TensorView v_cache,
      arena.Alloc(DataType::kBFloat16,
                  Shape({num_blocks, kPageSize, kKvHeads, kHeadDim})));
  INFERX_ASSIGN_OR_RETURN(
      const TensorView q,
      arena.Alloc(DataType::kBFloat16, Shape({batch, kQHeads, kHeadDim})));
  INFERX_ASSIGN_OR_RETURN(
      const TensorView out,
      arena.Alloc(DataType::kBFloat16, Shape({batch, kQHeads, kHeadDim})));

  // Blocks assigned back to front, as a real allocator would after churn.
  std::vector<int32_t> dense(static_cast<size_t>(batch * blocks_per_seq));
  std::vector<int32_t> blocks_used(static_cast<size_t>(batch),
                                   static_cast<int32_t>(blocks_per_seq));
  std::vector<int32_t> q_pos(static_cast<size_t>(batch),
                             static_cast<int32_t>(context - 1));
  std::vector<int32_t> seq_of_token(static_cast<size_t>(batch));
  std::vector<int32_t> last_page(static_cast<size_t>(batch));

  int32_t next = static_cast<int32_t>(num_blocks) - 1;
  for (int64_t s = 0; s < batch; ++s) {
    for (int64_t b = 0; b < blocks_per_seq; ++b) {
      dense[static_cast<size_t>(s * blocks_per_seq + b)] = next--;
    }
    seq_of_token[static_cast<size_t>(s)] = static_cast<int32_t>(s);
    const int64_t rem = context % kPageSize;
    last_page[static_cast<size_t>(s)] =
        static_cast<int32_t>(rem == 0 ? kPageSize : rem);
  }

  INFERX_ASSIGN_OR_RETURN(const TensorView dense_v, arena.Upload(dense));
  INFERX_ASSIGN_OR_RETURN(const TensorView seq_v, arena.Upload(seq_of_token));
  INFERX_ASSIGN_OR_RETURN(const TensorView pos_v, arena.Upload(q_pos));

  // The dense table has to be seen as [batch, blocks_per_seq].
  INFERX_ASSIGN_OR_RETURN(
      const TensorView table_v,
      TensorView::Create(dense_v.Data(), DataType::kInt32,
                         Shape({batch, blocks_per_seq}), DeviceId::Cuda(0)));

  std::vector<int32_t> indices, indptr;
  INFERX_RETURN_IF_ERROR(ops::BuildCsrBlockTable(
      dense, batch, blocks_per_seq, blocks_used, &indices, &indptr));

  INFERX_ASSIGN_OR_RETURN(const TensorView indices_v, arena.Upload(indices));
  INFERX_ASSIGN_OR_RETURN(const TensorView indptr_v, arena.Upload(indptr));
  INFERX_ASSIGN_OR_RETURN(const TensorView last_page_v,
                          arena.Upload(last_page));

  // Bytes of KV read per decode step. This is what the kernel is bound by: the
  // arithmetic is one query row against `context` keys, so the whole cost is
  // streaming K and V.
  const double kv_bytes = 2.0 * static_cast<double>(batch) * context *
                          kKvHeads * kHeadDim * 2.0;

  // --- ours ---------------------------------------------------------------
  // The naive kernel keeps one float of shared memory per key, so it simply
  // cannot run past ~12k context. Skipped rather than failed: that limit is a
  // known property of the reference, and it is most of why FlashInfer exists.
  double ours_ms = 0;
  *ours_ran = false;

  const size_t smem = (128u + static_cast<size_t>(blocks_per_seq * kPageSize)) *
                      sizeof(float);

  if (smem <= 48u * 1024u) {
    INFERX_ASSIGN_OR_RETURN(
        const Timing t,
        TimeLaunch(
            [&] {
              return ops::PagedAttention(q, k_cache, v_cache, table_v,
                                             seq_v, pos_v, out, scale);
            },
            warmup, iters));

    ours_ms = t.min_ms;
    *ours_ran = true;

    std::printf("%6ld %8ld  %-11s %9.4f %7.1f %9.1f\n", static_cast<long>(batch),
                static_cast<long>(context), "reference", t.min_ms,
                t.noise() * 100.0, t.gbytes_per_s(kv_bytes));
  } else {
    std::printf("%6ld %8ld  %-11s %9s %7s %9s   (needs %zu KB of shared "
                "memory)\n",
                static_cast<long>(batch), static_cast<long>(context),
                "reference", "-", "-", "-", smem / 1024);
  }

  // --- FlashInfer ----------------------------------------------------------
  INFERX_ASSIGN_OR_RETURN(
      const Timing t,
      TimeLaunch(
          [&] {
            return fi->Decode(q, k_cache, v_cache, indices_v, indptr_v,
                              absl::MakeConstSpan(indptr), last_page_v, out,
                              scale);
          },
          warmup, iters));

  std::printf("%6ld %8ld  %-11s %9.4f %7.1f %9.1f", static_cast<long>(batch),
              static_cast<long>(context), "flashinfer", t.min_ms,
              t.noise() * 100.0, t.gbytes_per_s(kv_bytes));

  if (*ours_ran && t.min_ms > 0) {
    std::printf("   %6.2fx\n", ours_ms / t.min_ms);
  } else {
    std::printf("\n");
  }

  return OkStatus();
}

int Main(int argc, char** argv) {
  int warmup = 10;
  int iters = 40;

  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--warmup" && i + 1 < argc) warmup = std::atoi(argv[++i]);
    else if (arg == "--iters" && i + 1 < argc) iters = std::atoi(argv[++i]);
    else {
      std::fprintf(stderr, "usage: %s [--warmup N] [--iters N]\n", argv[0]);
      return 2;
    }
  }

  if (!cuda::Available()) {
    std::fprintf(stderr, "no CUDA device available\n");
    return 1;
  }

  cudaDeviceProp prop{};
  if (cudaGetDeviceProperties(&prop, 0) == cudaSuccess) {
    std::printf("device: %s (sm_%d%d)\n", prop.name, prop.major, prop.minor);
  }

  std::printf("paged decode attention | q_heads %ld kv_heads %ld head_dim %ld "
              "page %ld\n",
              static_cast<long>(kQHeads), static_cast<long>(kKvHeads),
              static_cast<long>(kHeadDim), static_cast<long>(kPageSize));
  std::printf("one query token per sequence; GB/s is KV read per step\n\n");

  std::printf("%6s %8s  %-11s %9s %7s %9s   %s\n", "batch", "context", "kernel",
              "best_ms", "noise_%", "GB/s", "speedup");
  std::printf("%s\n", std::string(72, '-').c_str());

  auto fi = ops::FlashInferDecode::Create();
  if (!fi.ok()) {
    std::fprintf(stderr, "cannot create FlashInfer context: %s\n",
                 fi.status().ToString().c_str());
    return 1;
  }

  int failures = 0;

  for (const int64_t context : kContexts) {
    for (const int64_t batch : kBatches) {
      bool ours_ran = false;
      const Status s = RunOne(&*fi, batch, context, warmup, iters, &ours_ran);

      if (!s.ok()) {
        std::fprintf(stderr, "batch=%ld context=%ld FAILED: %s\n",
                     static_cast<long>(batch), static_cast<long>(context),
                     s.ToString().c_str());
        ++failures;
      }
    }
    std::printf("\n");
  }

  return failures == 0 ? 0 : 1;
}

}  // namespace
}  // namespace inferx::bench

int main(int argc, char** argv) { return inferx::bench::Main(argc, argv); }
