#pragma once

#include <array>
#include <atomic>
#include <cstdint>
#include <memory>
#include <vector>

#include "inferx/core/device.h"
#include "inferx/core/status.h"
#include "inferx/core/tensor_view.h"

namespace inferx::comm {

enum class CommBackend : uint8_t {
  kSingleRank,
  kHostSim,
  kNccl,
  kMscclpp,
};

struct CommCapabilities {
  bool cuda_graph_capture = false;
  bool device_collectives = false;
};

const char* CommBackendName(CommBackend backend);

class Communicator;

struct CommMetricSnapshot {
  uint64_t all_reduce_calls = 0;
  uint64_t all_reduce_bytes = 0;
  uint64_t collective_failures = 0;
  uint64_t aborts = 0;
  std::array<uint64_t, 13> latency_buckets{};
  uint64_t latency_count = 0;
  double latency_sum_seconds = 0.0;
  uint64_t timing_samples_dropped = 0;
  uint64_t timing_graph_skips = 0;
};

inline constexpr std::array<double, 13> kCollectiveLatencyBuckets = {
    .00001, .000025, .00005, .0001, .00025, .0005, .001,
    .0025,  .005,    .01,    .025,  .05,    .1};

struct CommObservationConfig {
  /// Zero disables CUDA-event timing. N samples every Nth collective.
  uint64_t timing_sample_every = 0;
  size_t timing_ring_size = 256;
};

/// Backend-neutral counters shared with the owner after the decorated
/// communicator itself has moved into a model.
class CommMetrics {
 public:
  CommMetricSnapshot Snapshot() const noexcept;
  void RecordAllReduce(uint64_t bytes, bool success) noexcept;
  void RecordAbort() noexcept;
  void RecordLatency(double seconds) noexcept;
  void RecordTimingDrop() noexcept;
  void RecordGraphSkip() noexcept;

 private:
  std::atomic<uint64_t> all_reduce_calls_{0};
  std::atomic<uint64_t> all_reduce_bytes_{0};
  std::atomic<uint64_t> collective_failures_{0};
  std::atomic<uint64_t> aborts_{0};
  std::array<std::atomic<uint64_t>, 13> latency_buckets_{};
  std::atomic<uint64_t> latency_count_{0};
  std::atomic<double> latency_sum_seconds_{0.0};
  std::atomic<uint64_t> timing_samples_dropped_{0};
  std::atomic<uint64_t> timing_graph_skips_{0};
};

/// Collective communication used by tensor-parallel model layers.
///
/// Every rank must issue collectives in identical order. This mirrors NCCL's
/// SPMD contract: rank-local control flow around a collective is a bug.
class Communicator {
 public:
  virtual ~Communicator() = default;

  virtual int rank() const = 0;
  virtual int size() const = 0;
  virtual DeviceId device() const = 0;
  virtual CommBackend backend() const = 0;
  virtual CommCapabilities capabilities() const = 0;

  /// Sums corresponding elements across ranks and writes the result in-place
  /// on every rank. All ranks must provide the same dtype and element count.
  /// `stream` is the backend's opaque execution stream (a cudaStream_t for
  /// CUDA/NCCL). Host tensors ignore it.
  virtual Status AllReduceSum(const TensorView& tensor,
                              void* stream = nullptr) = 0;

  /// Breaks outstanding backend work during fatal rank failure. Idempotent.
  virtual Status Abort() = 0;
};

/// Wraps any communicator with lock-free call/byte/error accounting. The
/// wrapper adds no CUDA operations and measures no host-call duration.
std::unique_ptr<Communicator> ObserveCommunicator(
    std::unique_ptr<Communicator> inner, std::shared_ptr<CommMetrics> metrics,
    CommObservationConfig config = {});

/// The production default at TP=1. The collective validates its input and is
/// otherwise a no-op, so model code always exercises the communicator call.
class SingleRankComm final : public Communicator {
 public:
  explicit SingleRankComm(DeviceId device = DeviceId::Cuda(0))
      : device_(device) {}
  int rank() const override { return 0; }
  int size() const override { return 1; }
  DeviceId device() const override { return device_; }
  CommBackend backend() const override { return CommBackend::kSingleRank; }
  CommCapabilities capabilities() const override {
    return {.cuda_graph_capture = true, .device_collectives = true};
  }
  Status AllReduceSum(const TensorView& tensor,
                      void* stream = nullptr) override;
  Status Abort() override { return OkStatus(); }

 private:
  DeviceId device_;
};

/// Creates one host-simulated communicator per rank, sharing one rendezvous.
///
/// HostSimComm accepts CPU tensors and, in a CUDA build, CUDA tensors staged
/// through host memory. Each rank is intended to run on its own host thread.
/// Reduction order is rank 0..N-1, making numerical tests reproducible without
/// multi-GPU hardware. CUDA staging is a correctness backend, not a benchmark.
StatusOr<std::vector<std::unique_ptr<Communicator>>> CreateHostSimCommunicators(
    int size);

}  // namespace inferx::comm
