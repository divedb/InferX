#pragma once

#include <concepts>
#include <string_view>

#include "inferx/models/components/attention.h"
#include "inferx/models/components/mlp.h"
#include "inferx/models/components/norm.h"
#include "inferx/models/loading/weight_names.h"

namespace inferx::causal {

enum class NormPlacement { kPre };

/// Architecture policy for a homogeneous dense decoder. Dimensions, epsilon,
/// RoPE parameters, and weight tying remain checkpoint configuration.
template <class T>
concept ModelTraits = requires {
  { T::kModelType } -> std::convertible_to<std::string_view>;
  { T::kArch } -> std::convertible_to<std::string_view>;
  { T::kNames } -> std::convertible_to<models::WeightNames>;
  { T::kLayout } -> std::convertible_to<models::WeightLayout>;
  { T::kNorm } -> std::convertible_to<NormPlacement>;
  typename T::Norm;
  typename T::Attn;
  typename T::Mlp;
};

}  // namespace inferx::causal
