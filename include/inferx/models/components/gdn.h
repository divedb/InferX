/// \file
/// \brief Gated DeltaNet linear attention (Qwen3-Next recurrent layers).

#ifndef INFERX_MODELS_COMPONENTS_GDN_H_
#define INFERX_MODELS_COMPONENTS_GDN_H_

#include <cstdint>
#include <optional>
#include <utility>

#include "inferx/cache/recurrent_state_pool.h"
#include "inferx/core/tensor.h"
#include "inferx/models/components/attention.h"
#include "inferx/models/state.h"
#include "inferx/ops/execution_context.h"

namespace inferx {

struct AttentionBatch;
class DiagnosticTrace;

namespace components {

/// Weights of one Gated DeltaNet mixer, as loaded.
struct GdnWeights {
  Tensor in_proj_qkvz;   ///< [2*kh*kd + 2*vh*vd, hidden]
  Tensor in_proj_ba;     ///< [2*vh, hidden], per-group [b | a]
  Tensor conv1d;         ///< [conv_dim, kernel], depthwise
  Tensor a_log;          ///< [vh] float32 discretization logs
  Tensor dt_bias;        ///< [vh] float32 time-step bias
  Tensor norm_weight;    ///< [vd] gated-norm weight
  Tensor out_proj;       ///< [hidden, vh * vd]
};

/// Reusable GDN workspace, sized once by the decoder stack.
struct GdnWorkspace {
  std::optional<Tensor> packed;    ///< [max_tokens, 2kd+2vd kh-groups]
  std::optional<Tensor> conv_in;   ///< [max_tokens, 2*kh*kd + vh*vd]
  std::optional<Tensor> z;         ///< [max_tokens, vh*vd]
  std::optional<Tensor> ba;        ///< [max_tokens, 2*vh]
  std::optional<Tensor> beta;      ///< [max_tokens, vh] float32
  std::optional<Tensor> g;         ///< [max_tokens, vh] float32
  std::optional<Tensor> y;         ///< [max_tokens, vh*vd]
};

/// \brief Runs one recurrent layer into `mixed_out` ([rows, hidden]).
///
/// Split the per-group projection, run the depthwise causal convolution
/// with its state ring, compute the gates, advance the delta-rule state
/// sequentially over the step's tokens, and close with the gated norm and
/// output projection.
Status RunGatedDeltaNet(const GatedDeltaNetConfig& config, const GdnWeights& weights,
                        const Tensor& normed, const AttentionBatch& batch,
                        const RecurrentState& state, const RecurrentStatePool& pool,
                        GdnWorkspace& ws, ops::ExecutionContext& ctx, DiagnosticTrace* trace,
                        std::string_view prefix, Tensor& mixed_out);

}  // namespace inferx::components
}  // namespace inferx

#endif  // INFERX_MODELS_COMPONENTS_GDN_H_
