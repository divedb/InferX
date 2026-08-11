// Where a decode step's time actually goes.
//
// The step measures 13.9 ms at batch 1 against an 8.4 ms weight-bandwidth
// floor, and the 5.5 ms gap has so far been attributed by guesswork. This
// attributes it by measurement: every component of one layer is timed at the
// decode shape, scaled by the layer count, and the total is compared against
// the step. Anything the sum does not explain is overhead between components.
//
// Written because the guess was wrong. The gap was blamed on GEMM efficiency at
// m=1, but bench/gemm_bench.cc had already measured those GEMMs at 645-696 GB/s
// -- 88-95% of this card's peak -- which leaves very little for a "fix" to
// recover. Attribution before optimization.

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "absl/types/span.h"
#include "bench/cuda_timer.h"
#include "inferx/backends/cuda/cuda_utils.h"
#include "inferx/core/device_buffer.h"
#include "inferx/core/shape.h"
#include "inferx/core/status.h"
#include "inferx/core/tensor_view.h"
#include "inferx/ops/flashinfer_attention.h"
#include "inferx/ops/gemm.h"
#include "inferx/ops/layers.h"

namespace inferx::bench {
namespace {

using bf16 = __nv_bfloat16;

// Qwen2.5-3B.
constexpr int64_t kLayers = 36;
constexpr int64_t kHidden = 2048;
constexpr int64_t kInter = 11008;
constexpr int64_t kQHeads = 16;
constexpr int64_t kKvHeads = 2;
constexpr int64_t kHeadDim = 128;
constexpr int64_t kVocab = 151936;
constexpr int64_t kPageSize = 16;
constexpr int64_t kContext = 256;
constexpr int64_t kBatch = 1;

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

  StatusOr<TensorView> I32(const std::vector<int32_t>& host) {
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

struct Row {
  const char* name;
  double per_call_ms;
  int64_t calls_per_step;
  double bytes_per_call;

  double total_ms() const { return per_call_ms * calls_per_step; }
};

int Main(int argc, char** argv) {
  int warmup = 10;
  int iters = 50;

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

  Arena a;
  std::vector<Row> rows;

  auto gemm = ops::CublasLtGemm::Create();
  if (!gemm.ok()) {
    std::fprintf(stderr, "%s\n", gemm.status().ToString().c_str());
    return 1;
  }

  const auto bf = [&](const Shape& s) {
    auto v = a.Alloc(DataType::kBFloat16, s);
    if (!v.ok()) std::fprintf(stderr, "alloc: %s\n",
                              v.status().ToString().c_str());
    return *v;
  };

  // Activations at the decode shape.
  const TensorView x = bf(Shape({kBatch, kHidden}));
  const TensorView qv = bf(Shape({kBatch, kQHeads * kHeadDim}));
  const TensorView kv = bf(Shape({kBatch, kKvHeads * kHeadDim}));
  const TensorView vv = bf(Shape({kBatch, kKvHeads * kHeadDim}));
  const TensorView av = bf(Shape({kBatch, kQHeads * kHeadDim}));
  const TensorView gate = bf(Shape({kBatch, kInter}));
  const TensorView up = bf(Shape({kBatch, kInter}));
  const TensorView logits = bf(Shape({kBatch, kVocab}));

  // Weights.
  const TensorView wqkv =
      bf(Shape({kQHeads * kHeadDim + 2 * kKvHeads * kHeadDim, kHidden}));
  const TensorView qkv_out =
      bf(Shape({kBatch, kQHeads * kHeadDim + 2 * kKvHeads * kHeadDim}));
  const TensorView wgate_up = bf(Shape({2 * kInter, kHidden}));
  const TensorView gate_up_out = bf(Shape({kBatch, 2 * kInter}));
  const TensorView wo = bf(Shape({kHidden, kQHeads * kHeadDim}));
  const TensorView wdown = bf(Shape({kHidden, kInter}));
  const TensorView wembed = bf(Shape({kVocab, kHidden}));
  const TensorView norm_w = bf(Shape({kHidden}));
  const TensorView bias_qkv =
      bf(Shape({kQHeads * kHeadDim + 2 * kKvHeads * kHeadDim}));

  const double w2 = 2.0;  // bf16

  const auto time_gemm = [&](const char* name, const TensorView& in,
                             const TensorView& w, const TensorView& out,
                             int64_t calls) {
    auto t = TimeLaunch([&] { return gemm->LinearBF16(in, w, out); }, warmup,
                        iters);
    if (!t.ok()) {
      std::fprintf(stderr, "%s: %s\n", name, t.status().ToString().c_str());
      return;
    }
    rows.push_back({name, t->min_ms, calls,
                    static_cast<double>(w.Numel()) * w2});
  };

  // Fused, matching what the model runs: q/k/v as one [q+2kv, hidden] GEMM and
  // gate/up as one [2*inter, hidden].
  time_gemm("qkv_proj (fused)", x, wqkv, qkv_out, kLayers);
  time_gemm("o_proj", av, wo, x, kLayers);
  time_gemm("gate_up (fused)", x, wgate_up, gate_up_out, kLayers);
  time_gemm("down_proj", gate, wdown, x, kLayers);
  time_gemm("lm_head", x, wembed, logits, 1);

  // --- attention ------------------------------------------------------------
  const int64_t blocks = (kContext + kPageSize - 1) / kPageSize;
  const TensorView k_cache =
      bf(Shape({blocks, kPageSize, kKvHeads, kHeadDim}));
  const TensorView v_cache =
      bf(Shape({blocks, kPageSize, kKvHeads, kHeadDim}));
  const TensorView q3 = bf(Shape({kBatch, kQHeads, kHeadDim}));
  const TensorView a3 = bf(Shape({kBatch, kQHeads, kHeadDim}));

  std::vector<int32_t> indices(static_cast<size_t>(blocks));
  for (int64_t i = 0; i < blocks; ++i) indices[static_cast<size_t>(i)] =
      static_cast<int32_t>(i);
  const std::vector<int32_t> indptr = {0, static_cast<int32_t>(blocks)};
  const std::vector<int32_t> last_page = {kPageSize};

  auto idx = a.I32(indices);
  auto ip = a.I32(indptr);
  auto lp = a.I32(last_page);

  auto fi = ops::FlashInferDecode::Create();
  if (!fi.ok()) {
    std::fprintf(stderr, "%s\n", fi.status().ToString().c_str());
    return 1;
  }

  const float scale = 1.0f / std::sqrt(static_cast<float>(kHeadDim));

  {
    auto t = TimeLaunch(
        [&] {
          return fi->Decode(q3, k_cache, v_cache, *idx, *ip,
                            absl::MakeConstSpan(indptr), *lp, a3, scale);
        },
        warmup, iters);
    if (t.ok()) {
      rows.push_back({"attention (flashinfer)", t->min_ms, kLayers,
                      2.0 * kContext * kKvHeads * kHeadDim * w2});
    }
  }

  // --- the small kernels ----------------------------------------------------
  const auto time_small = [&](const char* name, auto&& fn, int64_t calls) {
    auto t = TimeLaunch(fn, warmup, iters);
    if (t.ok()) rows.push_back({name, t->min_ms, calls, 0.0});
  };

  time_small("rms_norm", [&] { return ops::RmsNorm(x, norm_w, x, 1e-6f); },
             kLayers * 2 + 1);
  time_small("qkv split + bias", [&] {
    return ops::SplitQkvWithBias(qkv_out, bias_qkv, qv, kv, vv);
  }, kLayers);
  // Hoisted, deliberately. An earlier version allocated these inside the timed
  // lambda, so every iteration paid a cudaMalloc -- which is tens of
  // microseconds and swamped the kernels being measured. It inflated rope and
  // kv_append by roughly 8x and would have sent the whole investigation after
  // the wrong component.
  const TensorView k3 = bf(Shape({kBatch, kKvHeads, kHeadDim}));
  const TensorView v3 = bf(Shape({kBatch, kKvHeads, kHeadDim}));
  const TensorView pos1 = *a.I32({0});
  const TensorView slot1 = *a.I32({0});

  time_small("rope", [&] {
    return ops::RotaryEmbedding(q3, k3, pos1, 1e6f);
  }, kLayers);
  time_small("silu_mul (fused)", [&] {
    return ops::SiluMulFused(gate_up_out, gate);
  }, kLayers);
  time_small("residual_add", [&] { return ops::AddInPlace(x, x); },
             kLayers * 2);
  time_small("kv_append", [&] {
    return ops::AppendToKvCache(k3, v3, k_cache, v_cache, slot1);
  }, kLayers);

  // --- host-side per-step work ----------------------------------------------
  // Not kernels, but they sit on the critical path of every step: the planner
  // runs on the host, the index uploads are synchronous, and the logits come
  // back before the caller sees anything.
  {
    const std::vector<int32_t> plan_indptr = {0, static_cast<int32_t>(blocks)};

    for (int i = 0; i < 20; ++i) {
      (void)fi->Plan(kBatch, kQHeads, kKvHeads, kHeadDim, kPageSize,
                     absl::MakeConstSpan(plan_indptr), true);
    }
    cudaDeviceSynchronize();

    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; ++i) {
      (void)fi->Plan(kBatch, kQHeads, kKvHeads, kHeadDim, kPageSize,
                     absl::MakeConstSpan(plan_indptr), true);
    }
    cudaDeviceSynchronize();
    const auto t1 = std::chrono::steady_clock::now();

    rows.push_back({"flashinfer plan (host)",
                    std::chrono::duration<double, std::milli>(t1 - t0).count() /
                        iters,
                    1, 0.0});
  }

  {
    // Eight small pageable H2D copies per step, each synchronous.
    std::vector<int32_t> small(64, 0);
    auto scratch = a.Alloc(DataType::kInt32, Shape({64}));

    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; ++i) {
      for (int j = 0; j < 8; ++j) {
        cudaMemcpy(scratch->Data(), small.data(), small.size() * 4,
                   cudaMemcpyHostToDevice);
      }
    }
    const auto t1 = std::chrono::steady_clock::now();

    rows.push_back({"index uploads (pageable)",
                    std::chrono::duration<double, std::milli>(t1 - t0).count() /
                        iters,
                    1, 0.0});
  }

  {
    std::vector<uint16_t> host_logits(static_cast<size_t>(kVocab));

    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; ++i) {
      cudaMemcpy(host_logits.data(), logits.Data(), host_logits.size() * 2,
                 cudaMemcpyDeviceToHost);
    }
    const auto t1 = std::chrono::steady_clock::now();

    rows.push_back({"logits D2H (pageable)",
                    std::chrono::duration<double, std::milli>(t1 - t0).count() /
                        iters,
                    1, 0.0});
  }

  // --- report ---------------------------------------------------------------
  std::printf("Qwen2.5-3B decode step, batch %ld, context %ld\n\n",
              static_cast<long>(kBatch), static_cast<long>(kContext));
  std::printf("%-24s %10s %7s %11s %9s\n", "component", "per_call", "calls",
              "total_ms", "GB/s");
  std::printf("%s\n", std::string(66, '-').c_str());

  double sum = 0;
  for (const Row& r : rows) {
    sum += r.total_ms();
    if (r.bytes_per_call > 0) {
      std::printf("%-24s %9.4f %7ld %11.3f %9.1f\n", r.name, r.per_call_ms,
                  static_cast<long>(r.calls_per_step), r.total_ms(),
                  r.bytes_per_call / (r.per_call_ms * 1e-3) / 1e9);
    } else {
      std::printf("%-24s %9.4f %7ld %11.3f %9s\n", r.name, r.per_call_ms,
                  static_cast<long>(r.calls_per_step), r.total_ms(), "-");
    }
  }

  std::printf("%s\n", std::string(66, '-').c_str());
  std::printf("%-24s %9s %7s %11.3f\n", "sum of components", "", "", sum);

  const double weights_gb =
      (kLayers * (2.0 * kHidden * kHidden + 2.0 * kKvHeads * kHeadDim * kHidden +
                  3.0 * kInter * kHidden) +
       static_cast<double>(kVocab) * kHidden) * w2 / 1e9;

  std::printf("%-24s %9s %7s %11.3f   (%.2f GB at 736 GB/s)\n",
              "bandwidth floor", "", "", weights_gb / 736e9 * 1e12,  // GB -> ms at 736 GB/s; the
                                         // trailing /1e9 here printed 0.000
                                         // and hid the reference number the
                                         // whole progression is quoted against
              weights_gb);
  // No hardcoded step time. It changes with every optimization and a stale
  // constant here misattributes the difference to whatever was measured last;
  // run bench/decode_step_bench.cc for the current figure.

  return 0;
}

}  // namespace
}  // namespace inferx::bench

int main(int argc, char** argv) { return inferx::bench::Main(argc, argv); }
