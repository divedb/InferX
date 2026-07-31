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

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
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
#include "inferx/kernels/quantize.h"

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

// One measured row. `fp8` selects the M1 prototype path over the baseline; the
// two are deliberately run through the same allocation, fill, warm and timing
// code so that a difference between them is a difference between the kernels.
Status RunOne(kernels::CublasLtGemm& gemm, const GemmShape& shape, int64_t m,
              int warmup, int iters, bool fp8, double* out_tflops) {
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

  // FP8 needs the operands quantized and their scales resident. Done once, out
  // of the timed region: on the serving path weights are quantized at load and
  // activation scales come from the previous layer, so paying for it per launch
  // would measure a cost the engine will not have.
  DeviceBuffer xq_buf, wq_buf, scale_buf;
  TensorView xq, wq;
  float* x_scale = nullptr;
  float* w_scale = nullptr;

  if (fp8) {
    INFERX_ASSIGN_OR_RETURN(
        xq_buf, DeviceBuffer::Allocate(static_cast<size_t>(m * k),
                                       DeviceId::Cuda(0)));
    INFERX_ASSIGN_OR_RETURN(
        wq_buf, DeviceBuffer::Allocate(static_cast<size_t>(n * k),
                                       DeviceId::Cuda(0)));
    INFERX_ASSIGN_OR_RETURN(
        scale_buf,
        DeviceBuffer::Allocate(2 * sizeof(float), DeviceId::Cuda(0)));

    INFERX_ASSIGN_OR_RETURN(
        xq, TensorView::Create(xq_buf.data(), DataType::kFloat8E4M3FN,
                               Shape({m, k}), DeviceId::Cuda(0)));
    INFERX_ASSIGN_OR_RETURN(
        wq, TensorView::Create(wq_buf.data(), DataType::kFloat8E4M3FN,
                               Shape({n, k}), DeviceId::Cuda(0)));

    x_scale = reinterpret_cast<float*>(scale_buf.data());
    w_scale = x_scale + 1;

    INFERX_RETURN_IF_ERROR(kernels::ComputeF8Scale(x, x_scale));
    INFERX_RETURN_IF_ERROR(kernels::ComputeF8Scale(w, w_scale));
    INFERX_RETURN_IF_ERROR(kernels::QuantizeF16ToF8E4M3(x, xq, x_scale));
    INFERX_RETURN_IF_ERROR(kernels::QuantizeF16ToF8E4M3(w, wq, w_scale));
    INFERX_CUDA_RETURN_IF_ERROR(cudaDeviceSynchronize());
  }

  // Planned before timing so the heuristic search is not part of the first
  // measured iteration. This is what Warm() is for on the serving path too.
  INFERX_RETURN_IF_ERROR(gemm.Warm(m, n, k, fp8));

  INFERX_ASSIGN_OR_RETURN(
      Timing t, TimeLaunch(
                    [&] {
                      return fp8 ? gemm.LinearF8E4M3(xq, wq, y, x_scale,
                                                     w_scale)
                                 : gemm.LinearF16(x, w, y);
                    },
                    warmup, iters));

  const double flop = 2.0 * static_cast<double>(m) * static_cast<double>(n) *
                      static_cast<double>(k);

  // Bytes moved depends on the path: FP8 halves both operands but the output
  // stays f16. That asymmetry is the whole argument at small m, where the
  // weight read dominates and halving it is nearly a 2x.
  const double elem = fp8 ? 1.0 : 2.0;
  const double bytes = elem * (static_cast<double>(m) * k +
                               static_cast<double>(n) * k) +
                       2.0 * static_cast<double>(m) * n;

  // Arithmetic intensity decides the regime, and it is a property of the shape
  // rather than of the measurement: FLOP per byte moved, compared against the
  // card's own ratio of peak FLOP/s to peak bandwidth. Below that ratio the
  // shape cannot saturate the tensor cores no matter how good the kernel is.
  const char* bound = flop / bytes < 160.0 ? "memory" : "compute";

  if (out_tflops != nullptr) *out_tflops = t.tflops(flop);

  std::printf("%-10s %-5s %6ld %6ld %6ld %9.4f %7.1f %8.1f %8.1f  %s\n",
              shape.name, fp8 ? "fp8" : "f16", static_cast<long>(m),
              static_cast<long>(n), static_cast<long>(k), t.min_ms,
              t.noise() * 100.0, t.tflops(flop), t.gbytes_per_s(bytes), bound);
  return OkStatus();
}

// Drives the GPU hard for `seconds` before any measurement starts.
//
// OFF BY DEFAULT, and measured to be harmful on this card without a clock lock.
// The intent was to clear the cold-start clock ramp, but a consumer part under
// four seconds of full-tilt GEMM hits its power limit instead, and the
// memory-bound shapes then measure the throttled state: gate_up at m=1 fell
// from 696 GB/s to 310, and its FP8/FP16 ratio inverted to 0.73x. Unsoaked runs
// reproduce each other to three digits; soaked ones do not.
//
// Keep it for the case it was written for -- with `nvidia-smi -lgc` holding a
// sustainable clock, a soak makes the first measured shapes comparable to the
// last. Without a lock, leave it at zero.
Status WarmUpDevice(kernels::CublasLtGemm& gemm, double seconds) {
  constexpr int64_t m = 4096, n = 4096, k = 4096;

  INFERX_ASSIGN_OR_RETURN(
      DeviceBuffer a, DeviceBuffer::Allocate(
                          static_cast<size_t>(m * k) * sizeof(__half),
                          DeviceId::Cuda(0)));
  INFERX_ASSIGN_OR_RETURN(
      DeviceBuffer b, DeviceBuffer::Allocate(
                          static_cast<size_t>(n * k) * sizeof(__half),
                          DeviceId::Cuda(0)));
  INFERX_ASSIGN_OR_RETURN(
      DeviceBuffer c, DeviceBuffer::Allocate(
                          static_cast<size_t>(m * n) * sizeof(__half),
                          DeviceId::Cuda(0)));

  INFERX_RETURN_IF_ERROR(FillDevice(a, m * k, 0.3f));
  INFERX_RETURN_IF_ERROR(FillDevice(b, n * k, 0.9f));

  INFERX_ASSIGN_OR_RETURN(
      TensorView x, TensorView::Create(a.data(), DataType::kFloat16,
                                       Shape({m, k}), DeviceId::Cuda(0)));
  INFERX_ASSIGN_OR_RETURN(
      TensorView w, TensorView::Create(b.data(), DataType::kFloat16,
                                       Shape({n, k}), DeviceId::Cuda(0)));
  INFERX_ASSIGN_OR_RETURN(
      TensorView y, TensorView::Create(c.data(), DataType::kFloat16,
                                       Shape({m, n}), DeviceId::Cuda(0)));

  INFERX_RETURN_IF_ERROR(gemm.Warm(m, n, k));

  const auto start = std::chrono::steady_clock::now();

  while (std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
             .count() < seconds) {
    for (int i = 0; i < 20; ++i) INFERX_RETURN_IF_ERROR(gemm.LinearF16(x, w, y));
    INFERX_CUDA_RETURN_IF_ERROR(cudaDeviceSynchronize());
  }

  return OkStatus();
}

int ReportClockMhz() {
  int mhz = 0;
  cudaDeviceGetAttribute(&mhz, cudaDevAttrClockRate, 0);
  return mhz / 1000;
}

int Main(int argc, char** argv) {
  int warmup = 10;
  int iters = 50;
  double soak = 0.0;  // see WarmUpDevice: only useful with locked clocks

  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--warmup" && i + 1 < argc) warmup = std::atoi(argv[++i]);
    else if (arg == "--iters" && i + 1 < argc) iters = std::atoi(argv[++i]);
    else if (arg == "--soak" && i + 1 < argc) soak = std::atof(argv[++i]);
    else {
      std::fprintf(stderr,
                   "usage: %s [--warmup N] [--iters N] [--soak SECONDS]\n",
                   argv[0]);
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

  std::printf("%-10s %-5s %6s %6s %6s %9s %7s %8s %8s  %s\n", "gemm", "path",
              "m", "n", "k", "best_ms", "noise_%", "TFLOP/s", "GB/s", "bound");
  std::printf("%s\n", std::string(90, '-').c_str());

  auto gemm = kernels::CublasLtGemm::Create();
  if (!gemm.ok()) {
    std::fprintf(stderr, "failed to create cuBLASLt context: %s\n",
                 gemm.status().ToString().c_str());
    return 1;
  }

  if (soak > 0.0) {
    std::printf("soaking %.1fs to bring clocks up from idle...\n", soak);

    const Status s = WarmUpDevice(*gemm, soak);
    if (!s.ok()) {
      std::fprintf(stderr, "warm-up failed: %s\n", s.ToString().c_str());
      return 1;
    }

    std::printf("done (nominal clock %d MHz)\n\n", ReportClockMhz());
  }

  int failures = 0;
  double total_speedup = 0.0;
  int speedup_count = 0;

  for (const int64_t m : kTokenCounts) {
    for (const GemmShape& shape : kQwen7B) {
      double f16_tflops = 0.0;
      double f8_tflops = 0.0;

      for (const bool fp8 : {false, true}) {
        const Status s = RunOne(*gemm, shape, m, warmup, iters, fp8,
                                fp8 ? &f8_tflops : &f16_tflops);
        if (!s.ok()) {
          std::fprintf(stderr, "%-10s %s m=%ld FAILED: %s\n", shape.name,
                       fp8 ? "fp8" : "f16", static_cast<long>(m),
                       s.ToString().c_str());
          ++failures;
        }
      }

      if (f16_tflops > 0.0 && f8_tflops > 0.0) {
        const double speedup = f8_tflops / f16_tflops;
        std::printf("%-10s %-5s %6ld %6s %6s %9s %7s %8s %7.2fx\n", "",
                    "  ->", static_cast<long>(m), "", "", "", "", "", speedup);
        total_speedup += speedup;
        ++speedup_count;
      }
    }
    std::printf("\n");
  }

  std::printf("plans cached: %zu\n", gemm->PlanCacheSize());

  if (speedup_count > 0) {
    std::printf("mean fp8/f16 speedup across %d shapes: %.2fx\n", speedup_count,
                total_speedup / speedup_count);
  }

  return failures == 0 ? 0 : 1;
}

}  // namespace
}  // namespace inferx::bench

int main(int argc, char** argv) { return inferx::bench::Main(argc, argv); }
