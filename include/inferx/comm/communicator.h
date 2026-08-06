#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "inferx/core/status.h"
#include "inferx/core/tensor_view.h"

namespace inferx::comm {

/// Collective communication used by tensor-parallel model layers.
///
/// Every rank must issue collectives in identical order. This mirrors NCCL's
/// SPMD contract: rank-local control flow around a collective is a bug.
class Communicator {
 public:
  virtual ~Communicator() = default;

  virtual int rank() const = 0;
  virtual int size() const = 0;

  /// Sums corresponding elements across ranks and writes the result in-place
  /// on every rank. All ranks must provide the same dtype and element count.
  virtual Status AllReduceSum(const TensorView& tensor) = 0;
};

/// The production default at TP=1. The collective validates its input and is
/// otherwise a no-op, so model code always exercises the communicator call.
class SingleRankComm final : public Communicator {
 public:
  int rank() const override { return 0; }
  int size() const override { return 1; }
  Status AllReduceSum(const TensorView& tensor) override;
};

/// Creates one host-simulated communicator per rank, sharing one rendezvous.
///
/// HostSimComm accepts CPU f32, f64, i32, i64 and bf16 tensors. Each rank is
/// intended to run on its own host thread. Reduction order is rank 0..N-1,
/// making numerical tests reproducible without multi-GPU hardware.
StatusOr<std::vector<std::unique_ptr<Communicator>>>
CreateHostSimCommunicators(int size);

}  // namespace inferx::comm
