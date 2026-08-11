#pragma once

#include <memory>

#include "inferx/comm/communicator.h"
#include "inferx/core/device.h"

namespace inferx::engine {

// Placement and topology supplied to one model rank. The communicator is
// transferred to the rank model during construction; future model primitives
// can instead retain the context without changing the runner contract.
class ParallelContext {
 public:
  ParallelContext(DeviceId device, int tp_rank, int tp_size,
                  std::unique_ptr<comm::Communicator> tp_communicator)
      : device_(device),
        tp_rank_(tp_rank),
        tp_size_(tp_size),
        tp_communicator_(std::move(tp_communicator)) {}

  ParallelContext(ParallelContext&&) noexcept = default;
  ParallelContext& operator=(ParallelContext&&) noexcept = default;
  ParallelContext(const ParallelContext&) = delete;
  ParallelContext& operator=(const ParallelContext&) = delete;

  DeviceId device() const { return device_; }
  int tp_rank() const { return tp_rank_; }
  int tp_size() const { return tp_size_; }
  comm::Communicator& tp_comm() const { return *tp_communicator_; }

  std::unique_ptr<comm::Communicator> TakeTpCommunicator() {
    return std::move(tp_communicator_);
  }

 private:
  DeviceId device_;
  int tp_rank_ = 0;
  int tp_size_ = 1;
  std::unique_ptr<comm::Communicator> tp_communicator_;
};

}  // namespace inferx::engine
