// The FP16 baseline sweep.
//
// This produces the numbers M1's FP8 prototype has to beat. The shapes are the
// four GEMMs a Qwen2.5-7B issues, swept over token counts from decode (m=1) to
// a full prefill chunk (m=8192), because the answer changes completely across
// that range and a single shape would mislead.
//
// The column to watch is GB/s, not TFLOP/s. At m=1 the GEMM reads the entire
// weight matrix to produce one row of output, so it is bound by weight
// bandwidth and the arithmetic is nearly free -- which is the whole argument
// for quantizing weights, and is why FP8 is expected to pay off at decode
// before it pays off anywhere else. At m=8192 the same GEMM is compute-bound
// and quantization buys much less. Reading both columns is how you tell which
// regime a shape is in.

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "bench/cuda_timer.h"
#include "inferx/core/cuda_utils.h"
#include "inferx/core/device_buffer.h"
#include "inferx/core/shape.h"
#include "inferx/core/status.h"
#include "inferx/core/tensor_view.h"
#include "inferx/kernels/gemm.h"

namespace inferx::bench {
namespace {

struct GemmShape {
  const char* name;
  int64_t n;  // output features
  int64_t k;  // input features
};

// Qwen2.5-7B: hidden 3584, intermediate 18944, 28 q heads / 4 kv heads at
// head_dim 128. QKV and gate/up are fused, as they are in any serious
// implementation -- three separate small GEMMs would be three launches and
// three passes over the activations.
constexpr GemmShape kQwen7B[] = {
    {"qkv_proj", 4608, 3584},    // (28 + 4 + 4) * 128
    {"o_proj", 3584, 3584},
    {"gate_up", 37888, 3584},    // 2 * 18944
    {"down_proj", 3584, 18944},
};

// Decode through a full prefill chunk (max_num_batched_tokens defaults to 8192).
constexpr int64_t kTokenCounts[] = {1, 8, 32, 128, 512, 2048, 8192};

Status FillDevice(const DeviceBuffer& buf, int64_t elems, float salt) {
  std::vector<__half> host(static_cast<size_t>(elems));

  // Small values in a narrow band: f16 has 11 bits of mantissa, and a dot
  // product over k=18944 terms of uniform-magnitude values would saturate the
  // accumulator and make the timing depend on denormal handling.
  for (int64_t i = 0; i < elems; ++i) {
    host[static_cast<size_t>(i)] =
        __float2half(0.05f * std::sin(static_cast<float>(i) * 0.001f + salt));
  }

  INFERX_CUDA_RETURN_IF_ERROR(cudaMemcpy(buf.data(), host.data(),
                                         host.size() * sizeof(__half),
                                         cudaMemcpyHostToDevice));
  return OkStatus();
}

Status RunOne(kernels::CublasLtGemm& gemm, const GemmShape& shape, int64_t m,
              int warmup, int iters) {
  const int64_t n = shape.n;
  const int64_t k = shape.k;

  INFERX_ASSIGN_OR_RETURN(
      DeviceBuffer xb,
      DeviceBuffer::Allocate(static_cast<size_t>(m * k) * sizeof(__half),
                             DeviceId::Cuda(0)));
  INFERX_ASSIGN_OR_RETURN(
      DeviceBuffer wb,
      DeviceBuffer::Allocate(static_cast<size_t>(n * k) * sizeof(__half),
                             DeviceId::Cuda(0)));
  INFERX_ASSIGN_OR_RETURN(
      DeviceBuffer yb,
      DeviceBuffer::Allocate(static_cast<size_t>(m * n) * sizeof(__half),
                             DeviceId::Cuda(0)));

  INFERX_RETURN_IF_ERROR(FillDevice(xb, m * k, 0.0f));
  INFERX_RETURN_IF_ERROR(FillDevice(wb, n * k, 1.7f));

  INFERX_ASSIGN_OR_RETURN(
      TensorView x, TensorView::Create(xb.data(), DataType::kFloat16,
                                       Shape({m, k}), DeviceId::Cuda(0)));
  INFERX_ASSIGN_OR_RETURN(
      TensorView w, TensorView::Create(wb.data(), DataType::kFloat16,
                                       Shape({n, k}), DeviceId::Cuda(0)));
  INFERX_ASSIGN_OR_RETURN(
      TensorView y, TensorView::Create(yb.data(), DataType::kFloat16,
                                       Shape({m, n}), DeviceId::Cuda(0)));

  // Planned before timing so the heuristic search is not part of the first
  // measured iteration. This is what Warm() is for on the serving path too.
  INFERX_RETURN_IF_ERROR(gemm.Warm(m, n, k));

  INFERX_ASSIGN_OR_RETURN(
      Timing t,
      TimeLaunch([&] { return gemm.LinearF16(x, w, y); }, warmup, iters));

  const double flop = 2.0 * static_cast<double>(m) * static_cast<double>(n) *
                      static_cast<double>(k);
  const double bytes =
      2.0 * (static_cast<double>(m) * k + static_cast<double>(n) * k +
             static_cast<double>(m) * n);

  // Arithmetic intensity decides the regime, and it is a property of the shape
  // rather than of the measurement: FLOP per byte moved, compared against the
  // card's own ratio of peak FLOP/s to peak bandwidth. Below that ratio the
  // shape cannot saturate the tensor cores no matter how good the kernel is.
  const char* bound = flop / bytes < 160.0 ? "memory" : "compute";

  std::printf("%-10s %6ld %6ld %6ld %9.4f %9.4f %7.1f %8.1f %8.1f  %s\n",
              shape.name, static_cast<long>(m), static_cast<long>(n),
              static_cast<long>(k), t.min_ms, t.median_ms, t.noise() * 100.0,
              t.tflops(flop), t.gbytes_per_s(bytes), bound);
  return OkStatus();
}

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

  if (!CudaAvailable()) {
    std::fprintf(stderr, "no CUDA device available\n");
    return 1;
  }

  cudaDeviceProp prop{};
  if (cudaGetDeviceProperties(&prop, 0) == cudaSuccess) {
    std::printf("device: %s (sm_%d%d, %.1f GB)\n", prop.name, prop.major,
                prop.minor,
                static_cast<double>(prop.totalGlobalMem) / (1024.0 * 1024 * 1024));
  }

  std::printf("baseline: cuBLASLt f16 in, f32 accumulate | warmup %d, iters %d\n",
              warmup, iters);
  std::printf(
      "throughput is from the best sample; noise_%% is how far the median sits\n"
      "above it. Lock clocks (`nvidia-smi -lgc`) and close the desktop for\n"
      "numbers worth quoting.\n\n");

  // The caveat that most affects how these numbers should be read, and the
  // easiest one to forget: this card has 64 MB of L2, and every weight matrix
  // here is re-read by every iteration of the same shape. qkv_proj and o_proj
  // fit in L2 outright, so their m=1 rows report bandwidth above the card's
  // ~736 GB/s HBM peak -- they are not touching HBM at all. Real decode walks
  // every layer in turn and finds L2 cold, so treat the small-m rows as an
  // upper bound rather than as achievable decode throughput.
  std::printf(
      "NOTE: weights are re-read across iterations, so shapes that fit in L2\n"
      "      (64 MB) report above-HBM bandwidth. Real decode streams cold\n"
      "      weights per layer; small-m rows are an upper bound.\n\n");

  std::printf("%-10s %6s %6s %6s %9s %9s %7s %8s %8s  %s\n", "gemm", "m", "n",
              "k", "best_ms", "p50_ms", "noise_%", "TFLOP/s", "GB/s", "bound");
  std::printf("%s\n", std::string(88, '-').c_str());

  auto gemm = kernels::CublasLtGemm::Create();
  if (!gemm.ok()) {
    std::fprintf(stderr, "failed to create cuBLASLt context: %s\n",
                 gemm.status().ToString().c_str());
    return 1;
  }

  int failures = 0;

  for (const int64_t m : kTokenCounts) {
    for (const GemmShape& shape : kQwen7B) {
      const Status s = RunOne(*gemm, shape, m, warmup, iters);
      if (!s.ok()) {
        std::fprintf(stderr, "%-10s m=%ld FAILED: %s\n", shape.name,
                     static_cast<long>(m), s.ToString().c_str());
        ++failures;
      }
    }
    std::printf("\n");
  }

  std::printf("plans cached: %zu\n", gemm->PlanCacheSize());

  return failures == 0 ? 0 : 1;
}

}  // namespace
}  // namespace inferx::bench

int main(int argc, char** argv) { return inferx::bench::Main(argc, argv); }
