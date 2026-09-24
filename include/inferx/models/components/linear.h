/// \file
/// \brief One affine projection, as the checkpoint stores it.

#ifndef INFERX_MODELS_COMPONENTS_LINEAR_H_
#define INFERX_MODELS_COMPONENTS_LINEAR_H_

#include <optional>

#include "inferx/core/tensor.h"

namespace inferx::components {

/// \brief A projection's weight and optional bias.
struct LinearWeights {
  Tensor weight;                ///< [out, in]
  std::optional<Tensor> bias;   ///< [out]; absent when the projection is bias-free.
};

}  // namespace inferx::components

#endif  // INFERX_MODELS_COMPONENTS_LINEAR_H_
