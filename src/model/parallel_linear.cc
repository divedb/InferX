#include "inferx/model/parallel/linear.h"

namespace inferx::model::parallel {

Status ColumnParallelLinear::ForwardBf16(ops::CublasLtGemm& gemm,
                                         const TensorView& input,
                                         const TensorView& local_weight,
                                         const TensorView& local_output,
                                         Stream stream) {
  return gemm.LinearBF16(input, local_weight, local_output, stream);
}

Status RowParallelLinear::ForwardBf16(ops::CublasLtGemm& gemm,
                                      comm::Communicator& communicator,
                                      const TensorView& local_input,
                                      const TensorView& local_weight,
                                      const TensorView& output, Stream stream) {
  INFERX_RETURN_IF_ERROR(
      gemm.LinearBF16(local_input, local_weight, output, stream));
  return ReduceOutput(communicator, output, stream);
}

Status RowParallelLinear::ReduceOutput(comm::Communicator& communicator,
                                       const TensorView& output,
                                       Stream stream) {
  return communicator.AllReduceSum(output, stream);
}

}  // namespace inferx::model::parallel
