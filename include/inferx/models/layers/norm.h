#ifndef INFERX_MODELS_LAYERS_NORM_H_
#define INFERX_MODELS_LAYERS_NORM_H_

#include <optional>

#include "inferx/core/status.h"
#include "inferx/core/tensor.h"
#include "inferx/ops/norm.h"
#include "inferx/ops/op_context.h"

namespace inferx::models {

// One config type covers the RMS and LayerNorm flavors plus their scale and
// bias conventions; the layer wraps it instead of redefining it.
using ops::NormConfig;

/// \brief Learned tensors of one norm. `bias` is engaged iff the config's
///        `bias` is set.
struct NormWeights {
  Tensor weight;
  std::optional<Tensor> bias;
};

/// \brief Row normalization over the last dimension: RMS or LayerNorm per
///        the config, sharing one interface so decoder blocks select the
///        flavor as data.
class Norm {
 public:
  /// \brief Normalizes each row of `input` into `output`.
  ///
  /// `output` may alias `input`. Validation and backend dispatch live in
  /// the op entry points.
  Status Forward(ops::OpContext& ctx, const Tensor& input, Tensor& output) const;

  /// \brief Adds `input` into `residual` and normalizes the sum into
  ///        `output`.
  ///
  /// `residual` receives the dtype-rounded sum -- the sublayer output joins
  /// the residual stream here. `output` must be distinct from `residual`
  /// and may alias `input`. RMS runs the fused add+norm kernel, which on
  /// CUDA normalizes the FP32 sum before rounding; LayerNorm has no fused
  /// kernel and normalizes the rounded sum on every backend.
  Status AddForward(ops::OpContext& ctx, const Tensor& input, Tensor& residual,
                    Tensor& output) const;

  NormConfig config_;
  NormWeights weights_;
};

}  // namespace inferx::models

#endif  // INFERX_MODELS_LAYERS_NORM_H_
