/// \file
/// \brief Gated dense feed-forward: configuration, weights, and execution.

#ifndef INFERX_MODELS_COMPONENTS_MLP_H_
#define INFERX_MODELS_COMPONENTS_MLP_H_

#include <cstdint>
#include <optional>

#include "inferx/core/tensor.h"
#include "inferx/models/components/linear.h"
#include "inferx/ops/execution_context.h"

namespace inferx {

class DiagnosticTrace;

namespace components {

/// \brief Bias-free SwiGLU feed-forward.
struct SwiGluConfig {
  int64_t intermediate_size = 0;  ///< Gate/up width.
};

/// \brief SwiGLU projections for one expert or a dense layer.
///
/// `packed_gate_up` is the fused MergedColumnParallelLinear weight:
/// rank-local rows in block-contiguous order [gate | up]. The per-projection
/// weights are views into that allocation and share its storage. `down` is
/// the RowParallelLinear counterpart: `[hidden, rank_intermediate]`, whose
/// partial sums need cross-rank reduction once tensor parallelism runs.
struct SwiGluWeights {
  Tensor packed_gate_up;  ///< [2 * intermediate, hidden] fused rows.
  LinearWeights gate;     ///< View of packed_gate_up's gate rows.
  LinearWeights up;       ///< View of packed_gate_up's up rows.
  LinearWeights down;     ///< [hidden, intermediate]
};

/// \brief Reusable SwiGLU workspace, sized once by the decoder stack.
///
/// One buffer: the fused gate/up GEMM de-interleaves straight into `gate`
/// (PackedSiluAndMul writes [rows, intermediate]); the separate `up`
/// buffer of the unpacked path is gone with it.
struct MlpWorkspace {
  std::optional<Tensor> gate;  ///< [max_tokens * max_intermediate] flat.
};

/// \brief Runs the SwiGLU feed-forward and projects back into `mixed_out`
///        ([rows, hidden]).
///
/// One fused GEMM over the packed gate/up weight into `packed_buffer` (the
/// shared packed-projection workspace the stack always provides), then
/// PackedSiluAndMul, then the down projection.
Status RunSwiGlu(const SwiGluConfig& config, const SwiGluWeights& weights,
                 const Tensor& normed, MlpWorkspace& ws, Tensor* packed_buffer,
                 ops::ExecutionContext& ctx, DiagnosticTrace* trace,
                 std::string_view prefix, Tensor& mixed_out);

}  // namespace inferx::components
}  // namespace inferx

#endif  // INFERX_MODELS_COMPONENTS_MLP_H_
