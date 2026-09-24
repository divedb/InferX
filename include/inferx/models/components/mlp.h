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
struct SwiGluWeights {
  std::optional<Tensor> packed_gate_up;  ///< Concatenated gate/up rows, when packed.
  LinearWeights gate;  ///< [intermediate, hidden]
  LinearWeights up;    ///< [intermediate, hidden]
  LinearWeights down;  ///< [hidden, intermediate]
};

/// \brief Reusable SwiGLU workspace, sized once by the decoder stack.
struct MlpWorkspace {
  std::optional<Tensor> gate;  ///< [max_tokens * max_intermediate] flat.
  std::optional<Tensor> up;    ///< [max_tokens * max_intermediate] flat.
};

/// \brief Runs the SwiGLU feed-forward and projects back into `mixed_out`
///        ([rows, hidden]).
///
/// `packed_buffer` is the shared packed-projection workspace, or nullptr
/// when this layer's gate/up weights are not packed.
Status RunSwiGlu(const SwiGluConfig& config, const SwiGluWeights& weights,
                 const Tensor& normed, MlpWorkspace& ws, Tensor* packed_buffer,
                 ops::ExecutionContext& ctx, DiagnosticTrace* trace,
                 std::string_view prefix, Tensor& mixed_out);

}  // namespace inferx::components
}  // namespace inferx

#endif  // INFERX_MODELS_COMPONENTS_MLP_H_
