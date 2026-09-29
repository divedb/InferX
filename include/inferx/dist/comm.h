/// \file
/// \brief The per-rank communication seam for tensor-parallel workers.
///
/// Collectives sit behind this interface so the model path can express
/// "reduce my partial" without knowing whether the world is one process
/// (identity), one process hosting several virtual ranks (LoopbackComm,
/// the reference backend for sharded-execution tests), or a set of
/// worker processes (NcclComm over NCCL, one rank per device).

#ifndef INFERX_DIST_COMM_H_
#define INFERX_DIST_COMM_H_

#include <cstdint>
#include <limits>

#include "inferx/core/status.h"
#include "inferx/core/tensor.h"
#include "inferx/ops/execution_context.h"

namespace inferx {
namespace dist {

/// \brief Collectives over the worker world, from one worker's view.
///
/// Collectives use the context's stream. Success means work was submitted;
/// results are ready for subsequent work on that stream. Tensors and their
/// storage must remain alive until the stream completes. Submission may block
/// waiting for peers, so ranks must be driven concurrently.
///
/// Calls on each rank must be serialized, and all ranks must issue the same
/// collective sequence with matching input shapes and dtypes. Only BF16 is
/// supported today. All tensors must live on the context's device. A failed
/// collective invalidates the current forward; callers must not retry it.
class CommBackend {
 public:
  virtual ~CommBackend() = default;

  /// \brief Number of workers in the world.
  virtual int size() const = 0;
  /// \brief This worker's rank.
  virtual int rank() const = 0;

  /// \brief Elementwise sum of `partial` across all ranks, in place.
  ///
  /// Backed by all-reduce; row-parallel projections (o_proj, down_proj)
  /// produce rank-local partial sums that this folds into the full result.
  /// Also folds the masked partials of a vocab-parallel embedding lookup.
  /// Accepts any tensor rank, including scalars and empty tensors.
  virtual Status AllReduceSum(const ops::ExecutionContext& ctx, Tensor& partial) = 0;

  /// \brief Concatenates rank shards along the last dimension: each rank
  ///        contributes `partial` [rows, shard] and receives `full`
  ///        [rows, size * shard], in rank order. Both tensors must be BF16.
  /// Buffers must not overlap, except exact aliasing is allowed at size 1.
  virtual Status AllGatherLastDim(const ops::ExecutionContext& ctx, const Tensor& partial,
                                  Tensor& full) = 0;

 protected:
  static Status ValidateTensor(const ops::ExecutionContext& ctx, const Tensor& tensor) {
    if (tensor.GetDataType() != DataType::kBFloat16) {
      return InvalidArgumentError("collectives carry bfloat16, got ",
                                  DataTypeName(tensor.GetDataType()));
    }
    if (tensor.Device() != ctx.device()) {
      return InvalidArgumentError("collective tensors must live on the context's device");
    }
    return OkStatus();
  }

  Status ValidateGather(const ops::ExecutionContext& ctx, const Tensor& partial,
                        const Tensor& full) const {
    INFERX_RETURN_IF_ERROR(ValidateTensor(ctx, partial));
    INFERX_RETURN_IF_ERROR(ValidateTensor(ctx, full));
    if (size() <= 0 || partial.Rank() != 2 || full.Rank() != 2 ||
        partial.Dim(0) != full.Dim(0) ||
        partial.Dim(1) > std::numeric_limits<int64_t>::max() / size() ||
        full.Dim(1) != size() * partial.Dim(1)) {
      return InvalidArgumentError("all-gather shapes disagree: partial ",
                                  partial.GetShape().ToString(), " vs full ",
                                  full.GetShape().ToString(), " for world ", size());
    }
    if (partial.IsEmpty() || (size() == 1 && partial.Data() == full.Data())) {
      return OkStatus();
    }
    const auto input = reinterpret_cast<uintptr_t>(partial.Data());
    const auto output = reinterpret_cast<uintptr_t>(full.Data());
    const bool overlaps = input <= output
                              ? output - input < static_cast<uintptr_t>(partial.NBytes())
                              : input - output < static_cast<uintptr_t>(full.NBytes());
    if (overlaps) return InvalidArgumentError("all-gather buffers must not overlap");
    return OkStatus();
  }
};

/// \brief The world-of-one backend: every collective is the identity,
///        which is mathematically exact for tensor_parallel_size == 1.
class SingleRankComm final : public CommBackend {
 public:
  int size() const override { return 1; }
  int rank() const override { return 0; }

  Status AllReduceSum(const ops::ExecutionContext& ctx, Tensor& partial) override {
    return ValidateTensor(ctx, partial);
  }

  Status AllGatherLastDim(const ops::ExecutionContext& ctx, const Tensor& partial,
                          Tensor& full) override {
    INFERX_RETURN_IF_ERROR(ValidateGather(ctx, partial, full));
    if (full.Data() == partial.Data() || partial.IsEmpty()) return OkStatus();
    return ctx.runtime().CopyAsync(full.Data(), partial.Data(), partial.NBytes(),
                                   CopyKind::kDeviceToDevice, ctx.stream());
  }
};

}  // namespace dist
}  // namespace inferx

#endif  // INFERX_DIST_COMM_H_
