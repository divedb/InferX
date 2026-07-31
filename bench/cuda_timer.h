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

#include "inferx/core/cuda_utils.h"
#include "inferx/core/status.h"

namespace inferx::bench {

/// \brief The distribution of one measurement, in milliseconds of GPU time.
struct Timing {
  double median_ms = 0.0;
  double min_ms = 0.0;
  double max_ms = 0.0;
  int iterations = 0;

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
/// `launch` is invoked `warmup + iterations` times. The warmup passes are not
/// measured: they exist to pay for lazy module loading, the cuBLASLt plan, and
/// the clock ramp, none of which are properties of the kernel.
///
/// \tparam LaunchFn  Callable returning `Status`, enqueuing work and **not**
///                   synchronizing. A callable that synchronizes internally
///                   still measures correctly but serializes the iterations.
/// \param launch     The work to time.
/// \param warmup     Unmeasured passes before timing starts.
/// \param iterations Measured passes. The median is taken over these.
/// \return           The timing, or the first error `launch` reported.
template <typename LaunchFn>
StatusOr<Timing> TimeLaunch(LaunchFn&& launch, int warmup = 10,
                            int iterations = 50) {
  if (iterations <= 0) {
    return InvalidArgumentError("TimeLaunch needs iterations > 0, got ",
                                iterations);
  }

  for (int i = 0; i < warmup; ++i) INFERX_RETURN_IF_ERROR(launch());
  INFERX_CUDA_RETURN_IF_ERROR(cudaDeviceSynchronize());

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
    // Drained before each iteration so that every sample measures one launch
    // into an idle device. Without this the host runs ahead and the iterations
    // queue into each other: the event pair still brackets its own kernel, but
    // that kernel now contends with the tail of its predecessors, and the
    // samples stop being independent. It shows up as a spread of several
    // hundred percent, which is how this was found.
    //
    // The sync is outside the event pair, so its cost is not in the number.
    if (cudaDeviceSynchronize() != cudaSuccess) break;

    if (cudaEventRecord(starts[i]) != cudaSuccess) break;
    launch_status = launch();
    if (cudaEventRecord(stops[i]) != cudaSuccess) break;
  }

  const cudaError_t sync = cudaDeviceSynchronize();

  std::vector<double> ms;
  ms.reserve(iterations);

  if (launch_status.ok() && sync == cudaSuccess) {
    for (int i = 0; i < iterations; ++i) {
      float elapsed = 0.0f;
      if (cudaEventElapsedTime(&elapsed, starts[i], stops[i]) == cudaSuccess) {
        ms.push_back(static_cast<double>(elapsed));
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
  t.min_ms = ms.front();
  t.max_ms = ms.back();
  t.median_ms = ms[ms.size() / 2];

  return t;
}

}  // namespace inferx::bench
