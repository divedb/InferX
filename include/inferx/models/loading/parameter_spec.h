/// \file
/// \brief Parameter declarations for the generic weight loader: what shape
/// a mapped checkpoint tensor must have, and how its dimensions distribute
/// across tensor-parallel ranks. Shared layers declare their parameters in
/// these terms; the loader validates checkpoint tensors against the
/// declaration before upload.

#ifndef INFERX_MODELS_LOADING_PARAMETER_SPEC_H_
#define INFERX_MODELS_LOADING_PARAMETER_SPEC_H_

#include <string_view>

#include "inferx/core/device.h"
#include "inferx/core/shape.h"
#include "inferx/core/status.h"
#include "inferx/core/tensor.h"
#include "inferx/models/loading/checkpoint_reader.h"

namespace inferx::models::loading {

/// \brief How one parameter's dimensions distribute across ranks.
enum class ShardRule {
  /// Every rank holds the full tensor (norms, routers, latent tables).
  kReplicated,
  /// Dimension-0 row slices per rank (embedding rows; merged
  /// column-parallel gate|up and expert rows).
  kRowSharded,
  /// Dimension-1 column slices per rank (row-parallel inputs: o_proj,
  /// down projections).
  kColumnSharded,
  /// Head-derived row slices of the fused Q/K/V projections (packed
  /// [query (| gate) | key | value] rows).
  kQkvSharded,
};

/// \brief One mapped checkpoint tensor: its full logical shape and its
///        shard rule. `name` is the complete checkpoint tensor name,
///        suffix included.
struct ParameterSpec {
  std::string_view name;
  Shape shape;
  ShardRule shard = ShardRule::kReplicated;
};

/// \brief Loads one replicated parameter, shape-checked against its
///        declaration and uploaded as bfloat16. The workhorse for norms,
///        routers, and other full tensors every rank holds.
StatusOr<Tensor> LoadReplicatedParameter(const Checkpoint& checkpoint, const ParameterSpec& spec,
                                         DeviceId device);

}  // namespace inferx::models::loading

#endif  // INFERX_MODELS_LOADING_PARAMETER_SPEC_H_
