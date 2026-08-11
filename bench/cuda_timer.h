// CUDA-event timing for the benchmark harness.
//
// Wall-clock timing around a launch measures the launch, not the kernel: the
// call is asynchronous, so a host timer reports how long it took to enqueue
// work. CUDA events are recorded *in the stream*, so the interval between them
// is GPU time for the work between them, which is the number M1 needs.
//
// Timing is per iteration rather than one interval around a batch, so the
// result is a distribution rather than a mean. On a consumer card the mean is
// the least useful summary available -- boost clocks decay under sustained load
// and a single slow iteration from a clock transition drags it. The median is
// reported for that reason, and the spread alongside it so that an unstable
// measurement is visible instead of quietly wrong.
//
// For numbers worth comparing across runs, lock the clocks first:
//     sudo nvidia-smi -pm 1 && sudo nvidia-smi -lgc <mhz>
// Without that, expect several percent of drift between runs and do not read
// anything into differences smaller than that.

#pragma once

#include <cuda_runtime.h>

#include <algorithm>
#include <vector>

#include "inferx/backends/cuda/cuda_utils.h"
#include "inferx/core/status.h"

namespace inferx::bench {

/// \brief The distribution of one measurement, in milliseconds of GPU time.
struct Timing {
  double median_ms = 0.0;
  double min_ms = 0.0;
  double max_ms = 0.0;
  int iterations = 0;
  /// Launches per sample. Chosen automatically; reported so a suspiciously
  /// fast row can be checked against how many launches it was averaged over.
  int batch = 1;

  /// \brief How far the median sits above the best sample, as a fraction.
  ///
  /// Interference can only add time, so the gap between the best sample and the
  /// typical one is a measure of how much the machine intruded. Near zero means
  /// the number is solid; large means the box was busy and the comparison is
  /// only worth as much as this is small.
  double noise() const {
    return min_ms > 0.0 ? (median_ms - min_ms) / min_ms : 0.0;
  }

  /// \brief TFLOP/s from the best sample. \see BestIsTheHeadline.
  double tflops(double flop) const {
    return min_ms > 0.0 ? flop / (min_ms * 1e-3) / 1e12 : 0.0;
  }

  /// \brief GB/s from the best sample. \see BestIsTheHeadline.
  double gbytes_per_s(double bytes) const {
    return min_ms > 0.0 ? bytes / (min_ms * 1e-3) / 1e9 : 0.0;
  }

  // Why the best sample and not the median: we are comparing kernels, not
  // machines. Interference is strictly additive -- no scheduling accident makes
  // a kernel finish sooner than it can -- so the fastest sample is the closest
  // estimate of the kernel's actual cost and the most reproducible across runs.
  // The median is kept and printed beside it because a large gap means the box
  // was too busy for the measurement to be worth much, and that has to stay
  // visible rather than being smoothed away.
  struct BestIsTheHeadline {};
};

/// \brief Times `launch` on the default stream.
///
/// Each sample brackets a *batch* of back-to-back launches and divides, rather
/// than timing one launch. That is not an optimization, it is what makes short
/// kernels measurable at all: a decode-shaped GEMM runs in tens of
/// microseconds, and if the device is drained between every sample it idles
/// long enough to drop its clocks, so the next kernel starts slow. Measured
/// that way, a 17 us GEMM read as 113 us and the FP8/FP16 ratio inverted.
///
/// Batching keeps the device busy across the whole sample while the samples
/// stay independent of each other, and it also amortizes launch overhead the
/// way real back-to-back layer execution does.
///
/// The batch size is chosen automatically from a pilot measurement so that a
/// sample spans roughly a millisecond -- long enough to dwarf event overhead,
/// short enough that clocks do not drift within one.
///
/// \tparam LaunchFn Callable returning `Status`, enqueuing work and **not**
///                  synchronizing. One that synchronizes internally defeats the
///                  batching and will measure the idle-clock case instead.
/// \param launch    The work to time.
/// \param warmup    Unmeasured passes before timing starts.
/// \param samples   Number of measured batches. The statistics are over these.
/// \return          The timing, per launch, or the first error reported.
template <typename LaunchFn>
StatusOr<Timing> TimeLaunch(LaunchFn&& launch, int warmup = 10,
                            int samples = 50) {
  const int iterations = samples;

  if (iterations <= 0) {
    return InvalidArgumentError("TimeLaunch needs samples > 0, got ",
                                iterations);
  }

  for (int i = 0; i < warmup; ++i) INFERX_RETURN_IF_ERROR(launch());
  INFERX_CUDA_RETURN_IF_ERROR(cudaDeviceSynchronize());

  // Pilot: one timed launch, only to size the batch. Its own accuracy does not
  // matter -- being wrong by 2x picks a batch that is wrong by 2x, which still
  // lands the sample in the right order of magnitude.
  int batch = 1;
  {
    cudaEvent_t s0 = nullptr, s1 = nullptr;
    if (cudaEventCreate(&s0) == cudaSuccess &&
        cudaEventCreate(&s1) == cudaSuccess) {
      float pilot_ms = 0.0f;

      if (cudaEventRecord(s0) == cudaSuccess && launch().ok() &&
          cudaEventRecord(s1) == cudaSuccess &&
          cudaDeviceSynchronize() == cudaSuccess &&
          cudaEventElapsedTime(&pilot_ms, s0, s1) == cudaSuccess &&
          pilot_ms > 0.0f) {
        constexpr double kTargetMs = 1.0;
        const double want = kTargetMs / static_cast<double>(pilot_ms);
        batch = static_cast<int>(want < 1.0 ? 1.0 : (want > 512.0 ? 512.0
                                                                  : want));
      }
    }
    cudaEventDestroy(s0);
    cudaEventDestroy(s1);
  }

  std::vector<cudaEvent_t> starts(iterations, nullptr);
  std::vector<cudaEvent_t> stops(iterations, nullptr);

  // Created up front so that event creation -- which can allocate -- never
  // lands between a start and its stop.
  for (int i = 0; i < iterations; ++i) {
    INFERX_CUDA_RETURN_IF_ERROR(cudaEventCreate(&starts[i]));
    INFERX_CUDA_RETURN_IF_ERROR(cudaEventCreate(&stops[i]));
  }

  Status launch_status = OkStatus();

  for (int i = 0; i < iterations && launch_status.ok(); ++i) {
    // Drained between samples, not between launches. This is what keeps the
    // samples independent -- without it the host runs ahead and a sample
    // contends with the tail of its predecessors, which showed up as spreads
    // of several hundred percent. Inside a sample the launches stay
    // back-to-back so the device never goes idle mid-measurement.
    //
    // The sync is outside the event pair, so its cost is not in the number.
    if (cudaDeviceSynchronize() != cudaSuccess) break;

    if (cudaEventRecord(starts[i]) != cudaSuccess) break;

    for (int j = 0; j < batch && launch_status.ok(); ++j) {
      launch_status = launch();
    }

    if (cudaEventRecord(stops[i]) != cudaSuccess) break;
  }

  const cudaError_t sync = cudaDeviceSynchronize();

  std::vector<double> ms;
  ms.reserve(iterations);

  if (launch_status.ok() && sync == cudaSuccess) {
    for (int i = 0; i < iterations; ++i) {
      float elapsed = 0.0f;
      if (cudaEventElapsedTime(&elapsed, starts[i], stops[i]) == cudaSuccess) {
        // Per launch, not per batch: everything downstream -- TFLOP/s, GB/s,
        // the FP8/FP16 ratio -- is stated per launch.
        ms.push_back(static_cast<double>(elapsed) / batch);
      }
    }
  }

  // Destroyed unconditionally: every path above may have left events live, and
  // leaking them across a sweep of hundreds of shapes is its own problem.
  for (int i = 0; i < iterations; ++i) {
    cudaEventDestroy(starts[i]);
    cudaEventDestroy(stops[i]);
  }

  INFERX_RETURN_IF_ERROR(launch_status);
  INFERX_CUDA_RETURN_IF_ERROR(sync);

  if (ms.empty()) return InternalError("no timing samples were collected");

  std::sort(ms.begin(), ms.end());

  Timing t;
  t.iterations = static_cast<int>(ms.size());
  t.batch = batch;
  t.min_ms = ms.front();
  t.max_ms = ms.back();
  t.median_ms = ms[ms.size() / 2];

  return t;
}

}  // namespace inferx::bench
