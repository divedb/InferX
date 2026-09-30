/// \file
/// \brief NCCL-backed collectives for multi-process tensor parallelism.
///
/// libnccl is resolved at runtime with dlopen: builds need no NCCL
/// installation, hosts with one get real collectives. The controller
/// creates a unique id (NewUniqueId) and hands the same bytes to every
/// worker's Join -- over the existing worker IPC or an environment
/// variable; ranks then reduce and gather on their own CUDA streams.

#ifndef INFERX_DIST_NCCL_COMM_H_
#define INFERX_DIST_NCCL_COMM_H_

#include <array>
#include <cstdint>
#include <memory>
#include <vector>

#include "inferx/core/status.h"
#include "inferx/dist/comm.h"

namespace inferx {
namespace dist {

/// \brief NCCL's ncclUniqueId is 128 opaque bytes; the wire type for
///        bootstrap exchange.
using NcclUniqueId = std::array<std::uint8_t, 128>;

/// One process driving distinct GPUs. Host rendezvous validates each collective
/// before issuing grouped NCCL calls, so a rank failure can release its peers.
class NcclWorld {
 public:
  static StatusOr<std::shared_ptr<NcclWorld>> Create(const std::vector<DeviceId>& devices);
  std::unique_ptr<CommBackend> MakeRank(int rank);
  void Abort(const Status& reason);
  Status CheckHealth();

 private:
  explicit NcclWorld(std::shared_ptr<struct NcclWorldState> state);
  std::shared_ptr<struct NcclWorldState> state_;
};

class NcclComm final : public CommBackend {
 public:
  /// \brief Generates a fresh unique id on the controller. Fails with
  ///        NotFound when libnccl is unavailable.
  static StatusOr<NcclUniqueId> NewUniqueId();

  /// \brief Joins the world: every rank calls this with the SAME id, its
  ///        rank, and the world size; the call blocks until all ranks
  ///        joined. The communicator is bound to `device`; later contexts
  ///        must use that device. Fails with NotFound when libnccl is unavailable.
  static StatusOr<std::unique_ptr<NcclComm>> Join(int rank, int size, const NcclUniqueId& id,
                                                  DeviceId device);

  ~NcclComm() override;

  int size() const override { return size_; }
  int rank() const override { return rank_; }

  Status AllReduceSum(const ops::OpContext& ctx, Tensor& partial) override;
  Status AllGatherLastDim(const ops::OpContext& ctx, const Tensor& partial,
                          Tensor& full) override;

 private:
  NcclComm(int rank, int size, void* comm, DeviceId device);

  int rank_;
  int size_;
  void* comm_;  ///< ncclComm_t.
  DeviceId device_;
};

}  // namespace dist
}  // namespace inferx

#endif  // INFERX_DIST_NCCL_COMM_H_
