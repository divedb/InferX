#pragma once

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
  Status AllReduceSum(const TensorView& tensor, void* stream = nullptr) override;
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
StatusOr<std::vector<std::unique_ptr<Communicator>>>
CreateHostSimCommunicators(int size);

}  // namespace inferx::comm
