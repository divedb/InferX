/// \file
/// \brief Rotary position embedding parameters.

#ifndef INFERX_MODELS_COMPONENTS_ROPE_H_
#define INFERX_MODELS_COMPONENTS_ROPE_H_

#include <cstdint>
#include <string>

namespace inferx::components {

/// \brief Rotary position embedding applied inside an attention layer.
struct RotaryConfig {
  int64_t dim = 0;               ///< Rotated columns per head; even, <= head_dim.
  float theta = 10000.0f;        ///< Base frequency.
  float factor = 1.0f;           ///< Scaling factor applied by the flavor.
  std::string type = "default";  ///< Flavor: default, yarn, linear, ...
  /// Raw scaling JSON for flavors beyond the shared fields.
  std::string parameters_json = "{}";
};

}  // namespace inferx::components

#endif  // INFERX_MODELS_COMPONENTS_ROPE_H_
