/// \file
/// \brief The per-rank communication seam for tensor-parallel workers.
///
/// Collectives sit behind this interface so the model path can express
/// "reduce my partial" without knowing whether the world is one process
/// (identity), one node (in-memory or NCCL), or many. SingleRankComm is the
/// only backend today: with a world of one every collective is the identity
/// function, which is exactly correct -- and it is the only backend this
/// build can honor, since no NCCL library is present. The NCCL backend for
/// multi-GPU worlds arrives with the tensor-parallel milestone: same
/// interface, worker-per-device processes, one communicator per rank.

#ifndef INFERX_DIST_COMM_H_
#define INFERX_DIST_COMM_H_

#include "inferx/config/parallel_config.h"
#include "inferx/core/status.h"
#include "inferx/core/tensor.h"

namespace inferx {
namespace dist {

/// \brief Collectives over the worker world, from one worker's view.
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
  virtual Status AllReduceSumBf16(Tensor& partial) = 0;

  /// \brief Concatenates rank shards along the last dimension: each rank
  ///        contributes `partial` [rows, shard] and receives `full`
  ///        [rows, size * shard]. Vocab-parallel logits before sampling.
  virtual Status AllGatherLastDim(const Tensor& partial, Tensor& full) = 0;

  /// \brief Orders every rank past this point.
  virtual Status Barrier() = 0;
};

/// \brief The world-of-one backend: every collective is the identity,
///        which is mathematically exact for tensor_parallel_size == 1.
class SingleRankComm final : public CommBackend {
 public:
  explicit SingleRankComm(const ParallelConfig& parallel)
      : size_(parallel.tensor_parallel_size), rank_(parallel.tensor_parallel_rank) {}

  int size() const override { return size_; }
  int rank() const override { return rank_; }

  Status AllReduceSumBf16(Tensor& partial) override {
    return ValidateAndCopy(partial, partial);
  }

  Status AllGatherLastDim(const Tensor& partial, Tensor& full) override {
    return ValidateAndCopy(partial, full);
  }

  Status Barrier() override { return OkStatus(); }

 private:
  static Status ValidateAndCopy(const Tensor& partial, Tensor& out) {
    if (partial.GetDataType() != DataType::kBFloat16) {
      return InvalidArgumentError("collectives carry bfloat16, got ",
                                  DataTypeName(partial.GetDataType()));
    }
    if (out.Data() != partial.Data()) return partial.CopyTo(out);
    return OkStatus();
  }

  int size_;
  int rank_;
};

}  // namespace inferx::dist
}  // namespace inferx

#endif  // INFERX_DIST_COMM_H_
