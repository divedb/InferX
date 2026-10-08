/// \file
/// \brief Gated DeltaNet linear attention (Qwen3-Next recurrent layers):
/// per-group projections, a depthwise causal convolution with its state
/// ring, delta-rule recurrence, and a gated output norm.

#ifndef INFERX_MODELS_LAYERS_GATED_DELTA_NET_H_
#define INFERX_MODELS_LAYERS_GATED_DELTA_NET_H_

#include <cstdint>
#include <optional>
#include <utility>

#include "inferx/cache/recurrent_state_pool.h"
#include "inferx/core/status.h"
#include "inferx/core/tensor.h"
#include "inferx/models/model.h"
#include "inferx/models/requirements.h"
#include "inferx/ops/op_context.h"

namespace inferx::models::layers {

/// \brief Options of one Gated DeltaNet mixer layer.
struct GatedDeltaNetConfig {
  int64_t key_heads = 0;
  int64_t value_heads = 0;
  int64_t key_dim = 0;
  int64_t value_dim = 0;
  int64_t conv_kernel_size = 0;  ///< Depthwise conv kernel width.
};

/// \brief Weights of one Gated DeltaNet mixer, as loaded. (`kh/vh/kd/vd`
///        refer to GatedDeltaNetConfig fields; `hidden` is the model hidden
///        size.)
struct GdnWeights {
  Tensor in_proj_qkvz;  ///< [2*kh*kd + 2*vh*vd, hidden]
  Tensor in_proj_ba;    ///< [2*vh, hidden], per-group [b | a]
  Tensor conv1d;        ///< [conv_dim, kernel], depthwise
  Tensor a_log;         ///< [vh] float32 discretization logs
  Tensor dt_bias;       ///< [vh] float32 time-step bias
  Tensor norm_weight;   ///< [vd] gated-norm weight
  Tensor out_proj;      ///< [hidden, vh * vd]
};

/// \brief GDN workspace, sized once by the decoder stack.
struct GdnWorkspace {
  std::optional<Tensor> packed;  ///< [max_tokens, 2kd+2vd kh-groups]
  std::optional<Tensor> conv_in; ///< [max_tokens, 2*kh*kd + vh*vd]
  std::optional<Tensor> z;       ///< [max_tokens, vh*vd]
  std::optional<Tensor> ba;      ///< [max_tokens, 2*vh]
  std::optional<Tensor> beta;    ///< [max_tokens, vh] float32
  std::optional<Tensor> g;       ///< [max_tokens, vh] float32
  std::optional<Tensor> y;       ///< [max_tokens, vh*vd]
};

/// \brief Runs one recurrent layer into `mixed_out` ([rows, hidden]).
///
/// Pipeline: split the per-group projection, run the depthwise causal
/// convolution with its state ring, compute the gates, advance the
/// delta-rule state sequentially over the step's tokens, close with the
/// gated norm and output projection.
Status RunGatedDeltaNet(const GatedDeltaNetConfig& config, const GdnWeights& weights,
                        const Tensor& normed, const AttentionBatch& batch,
                        const RecurrentState& state, const RecurrentStatePool& pool,
                        GdnWorkspace& ws, ops::OpContext& ctx, Tensor& mixed_out);

}  // namespace inferx::models::layers

#endif  // INFERX_MODELS_LAYERS_GATED_DELTA_NET_H_
