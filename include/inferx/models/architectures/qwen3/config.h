/// \file
/// \brief Qwen3 dense configuration. The authoritative representation
/// throughout construction: parsed, validated, and owned here; different
/// checkpoint sizes instantiate the same architecture with different
/// values.

#ifndef INFERX_MODELS_ARCHITECTURES_QWEN3_CONFIG_H_
#define INFERX_MODELS_ARCHITECTURES_QWEN3_CONFIG_H_

#include "inferx/core/status.h"
#include "inferx/models/configs/dims.h"
#include "inferx/models/configs/rope.h"
#include "inferx/util/config_view.h"

namespace inferx::models::qwen3 {

struct Qwen3Config : ModelDims {
  RopeConfig rope;

  /// \brief Parses Qwen3 dimensions and rope settings. Missing required
  ///        fields error with the full field path; unsupported
  ///        behavior-changing keys fail here, before weight allocation.
  static StatusOr<Qwen3Config> Parse(const ConfigView& view);

  /// \brief Checks dimensions and rope geometry.
  Status Validate() const;
};

}  // namespace inferx::models::qwen3

#endif  // INFERX_MODELS_ARCHITECTURES_QWEN3_CONFIG_H_
