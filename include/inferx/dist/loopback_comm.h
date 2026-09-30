/// \file
/// \brief The in-process tensor-parallel world: LoopbackComm.
///
/// One LoopbackWorld hosts N virtual ranks in one process, on one device.
/// Ranks publish collective contributions into world-owned slot tensors and
/// meet at generation barriers; the sums and gathers run as ordinary device
/// ops on each caller's stream. This is the reference backend: it exercises
/// the exact collective placement of a sharded forward without a second
/// process or GPU, so tests can check TP=N logits against TP=1. Real
/// multi-device worlds use NcclComm.

#ifndef INFERX_DIST_LOOPBACK_COMM_H_
#define INFERX_DIST_LOOPBACK_COMM_H_

#include <cstdint>
#include <map>
#include <memory>
#include <tuple>
#include <vector>

#include "inferx/core/device.h"
#include "inferx/core/status.h"
#include "inferx/core/tensor.h"
#include "inferx/dist/comm.h"

namespace inferx {
namespace dist {

/// \brief N loopback ranks sharing one device; create once, hand each rank
///        its CommBackend.
class LoopbackWorld {
 public:
  /// \brief Creates a world of `size` ranks, driven by concurrent callers.
  static StatusOr<std::unique_ptr<LoopbackWorld>> Create(int size);

  /// \brief Rank `rank`'s collective backend; each rank gets its own.
  CommBackend& rank(int rank);
  /// Transfers a rank to a runner; rank(rank) must not be used afterwards.
  std::unique_ptr<CommBackend> TakeRank(int rank);

  int size() const { return size_; }
  /// Cancels a failed forward, including peers waiting inside a collective.
  void Abort(const Status& reason);

 private:
  LoopbackWorld(int size, std::shared_ptr<struct LoopbackState> state);

  int size_;
  std::vector<std::unique_ptr<class LoopbackComm>> ranks_;
  std::shared_ptr<struct LoopbackState> state_;

  friend class LoopbackComm;
};

/// \brief One rank's view of a LoopbackWorld.
///
/// Every collective follows publish -> synchronize -> barrier -> combine ->
/// synchronize -> barrier: slots are only read once every rank's
/// contribution has EXECUTED (not merely enqueued), and only reused once
/// every rank is done reading. Ranks must issue the same sequence of
/// collectives -- a sharded model forward does by construction.
class LoopbackComm final : public CommBackend {
 public:
  int size() const override;
  int rank() const override;

  Status AllReduceSum(const ops::OpContext& ctx, Tensor& partial) override;
  Status AllGatherLastDim(const ops::OpContext& ctx, const Tensor& partial,
                          Tensor& full) override;

  ~LoopbackComm() override;

 private:
  friend class LoopbackWorld;
  LoopbackComm(int rank, std::shared_ptr<struct LoopbackState> state);

  /// \brief This rank's contribution slot of the given shape, on the
  ///        context device; one slot per distinct shape keeps embedding,
  ///        hidden, and logit collectives from reallocating.
  StatusOr<Tensor*> Slot(const Shape& shape, DeviceId device);

  Status Publish(const ops::OpContext& ctx, const Tensor& contribution);
  Status Meet();                 ///< Barrier over the shared generation.
  Status Finish(Status status);  ///< Propagates local failures to waiting peers.

  int rank_;
  std::shared_ptr<struct LoopbackState> state_;
  std::map<std::tuple<DeviceKind, int, std::vector<int64_t>>, Tensor> slots_;
};

}  // namespace dist
}  // namespace inferx

#endif  // INFERX_DIST_LOOPBACK_COMM_H_
