#include "inferx/dist/loopback_comm.h"

#include <condition_variable>
#include <mutex>
#include <utility>
#include <vector>

#include "inferx/ops/elementwise.h"
#include "inferx/ops/gather.h"

namespace inferx {
namespace dist {

/// Shared synchronization and contribution exchange across the world.
struct LoopbackState {
  std::mutex mu;
  std::condition_variable cv;
  int arrived = 0;                       ///< Ranks inside the current barrier.
  int generation = 0;                    ///< Completed barrier count.
  std::vector<const Tensor*> published;  ///< Rank slots registered this round.
  Status failure;

  /// Blocks until all `size` ranks reach the barrier; the last one in
  /// advances the generation and wakes everyone.
  Status Wait(int size) {
    std::unique_lock<std::mutex> lock(mu);
    if (!failure.ok()) return failure;
    const int target = generation + 1;
    if (++arrived == size) {
      ++generation;
      arrived = 0;
      cv.notify_all();
      return OkStatus();
    }
    cv.wait(lock, [&] { return generation >= target || !failure.ok(); });
    return failure;
  }
};

StatusOr<std::unique_ptr<LoopbackWorld>> LoopbackWorld::Create(int size) {
  if (size <= 0) {
    return InvalidArgumentError("loopback world needs a positive size, got ", size);
  }
  auto state = std::make_shared<LoopbackState>();
  state->published.resize(size, nullptr);
  return std::unique_ptr<LoopbackWorld>(new LoopbackWorld(size, std::move(state)));
}

LoopbackWorld::LoopbackWorld(int size, std::shared_ptr<LoopbackState> state)
    : size_(size), state_(std::move(state)) {
  ranks_.reserve(size_);
  for (int r = 0; r < size_; ++r) {
    ranks_.push_back(std::unique_ptr<LoopbackComm>(new LoopbackComm(r, state_)));
  }
}

CommBackend& LoopbackWorld::rank(int rank) {
  // Creation validated the size; the reference is stable for the world's
  // lifetime because ranks_ never grows afterwards.
  return *ranks_[static_cast<size_t>(rank)];
}

void LoopbackWorld::Abort(const Status& reason) {
  const std::lock_guard<std::mutex> lock(state_->mu);
  if (state_->failure.ok()) {
    state_->failure = reason.ok() ? absl::CancelledError("loopback world aborted") : reason;
    state_->cv.notify_all();
  }
}

std::unique_ptr<CommBackend> LoopbackWorld::TakeRank(int rank) {
  return std::move(ranks_.at(static_cast<size_t>(rank)));
}

LoopbackComm::LoopbackComm(int rank, std::shared_ptr<LoopbackState> state)
    : rank_(rank), state_(std::move(state)) {}

LoopbackComm::~LoopbackComm() = default;

int LoopbackComm::size() const { return static_cast<int>(state_->published.size()); }
int LoopbackComm::rank() const { return rank_; }

StatusOr<Tensor*> LoopbackComm::Slot(const Shape& shape, DeviceId device) {
  const auto key = std::make_tuple(device.kind, static_cast<int>(device.index),
                                   std::vector<int64_t>(shape.begin(), shape.end()));
  auto found = slots_.find(key);
  if (found == slots_.end()) {
    INFERX_ASSIGN_OR_RETURN(auto slot, Tensor::Empty(DataType::kBFloat16, shape, device));
    found = slots_.emplace(key, std::move(slot)).first;
  }
  return &found->second;
}

Status LoopbackComm::Publish(const ops::ExecutionContext& ctx, const Tensor& contribution) {
  INFERX_RETURN_IF_ERROR(ValidateTensor(ctx, contribution));
  INFERX_ASSIGN_OR_RETURN(Tensor * slot, Slot(contribution.GetShape(), ctx.device()));
  if (!contribution.IsEmpty() && slot->Data() != contribution.Data()) {
    INFERX_RETURN_IF_ERROR(ctx.runtime().CopyAsync(slot->Data(), contribution.Data(),
                                                   contribution.NBytes(),
                                                   CopyKind::kDeviceToDevice, ctx.stream()));
  }
  {
    const std::lock_guard<std::mutex> lock(state_->mu);
    state_->published[static_cast<size_t>(rank_)] = slot;
  }
  // The slot must hold EXECUTED data before peers read it; stream order
  // alone does not cross streams.
  return ctx.runtime().SynchronizeStream(ctx.stream());
}

Status LoopbackComm::Meet() { return state_->Wait(size()); }

Status LoopbackComm::Finish(Status status) {
  const std::lock_guard<std::mutex> lock(state_->mu);
  if (!status.ok() && state_->failure.ok()) {
    state_->failure = std::move(status);
    state_->cv.notify_all();
  }
  return state_->failure;
}

Status LoopbackComm::AllReduceSum(const ops::ExecutionContext& ctx, Tensor& partial) {
  return Finish([&]() -> Status {
    INFERX_RETURN_IF_ERROR(Publish(ctx, partial));
    INFERX_RETURN_IF_ERROR(Meet());
    const auto& published = state_->published;
    for (const Tensor* peer : published) {
      if (peer->GetShape() != partial.GetShape() || peer->Device() != ctx.device()) {
        return InvalidArgumentError("all-reduce peers must have matching shapes and devices");
      }
    }
    // Every rank uses the same summation order, including BF16 rounding.
    if (!partial.IsEmpty()) {
      INFERX_RETURN_IF_ERROR(ctx.runtime().CopyAsync(partial.Data(), published.front()->Data(),
                                                     partial.NBytes(),
                                                     CopyKind::kDeviceToDevice, ctx.stream()));
    }
    auto op_ctx = ctx;
    for (int r = 1; r < size(); ++r) {
      INFERX_RETURN_IF_ERROR(ops::Add(op_ctx, partial, *published[r], partial));
    }
    INFERX_RETURN_IF_ERROR(ctx.runtime().SynchronizeStream(ctx.stream()));
    return Meet();
  }());
}

Status LoopbackComm::AllGatherLastDim(const ops::ExecutionContext& ctx, const Tensor& partial,
                                      Tensor& full) {
  return Finish([&]() -> Status {
    INFERX_RETURN_IF_ERROR(ValidateGather(ctx, partial, full));
    INFERX_RETURN_IF_ERROR(Publish(ctx, partial));
    INFERX_RETURN_IF_ERROR(Meet());
    const auto& published = state_->published;
    auto op_ctx = ctx;
    for (int r = 0; r < size(); ++r) {
      if (published[r]->GetShape() != partial.GetShape() ||
          published[r]->Device() != ctx.device()) {
        return InvalidArgumentError("all-gather peers must have matching shapes and devices");
      }
      INFERX_RETURN_IF_ERROR(ops::CopyColumnBlock(op_ctx, *published[r], full,
                                                  static_cast<int64_t>(r) * partial.Dim(1)));
    }
    INFERX_RETURN_IF_ERROR(ctx.runtime().SynchronizeStream(ctx.stream()));
    return Meet();
  }());
}

}  // namespace dist
}  // namespace inferx
