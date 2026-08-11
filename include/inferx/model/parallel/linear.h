#pragma once

#include "inferx/comm/communicator.h"
#include "inferx/core/status.h"
#include "inferx/core/stream.h"
#include "inferx/core/tensor_view.h"
#include "inferx/ops/gemm.h"

namespace inferx::model::parallel {

// A linear whose output-feature dimension is partitioned across TP ranks.
// Its output stays rank-local, so completing the primitive requires no
// collective.
class ColumnParallelLinear {
 public:
  static Status ForwardBf16(ops::CublasLtGemm& gemm, const TensorView& input,
                            const TensorView& local_weight,
                            const TensorView& local_output, Stream stream = {});
};

// A linear whose input-feature dimension is partitioned across TP ranks.
// The local GEMM produces one partial full-width output; completing the
// primitive sums those partials on every rank.
class RowParallelLinear {
 public:
  static Status ForwardBf16(ops::CublasLtGemm& gemm,
                            comm::Communicator& communicator,
                            const TensorView& local_input,
                            const TensorView& local_weight,
                            const TensorView& output, Stream stream = {});

  // Completion seam for callers that select a quantized local GEMM. Keeping
  // the collective here preserves one communication boundary for every
  // storage format.
  static Status ReduceOutput(comm::Communicator& communicator,
                             const TensorView& output, Stream stream = {});
};

}  // namespace inferx::model::parallel
