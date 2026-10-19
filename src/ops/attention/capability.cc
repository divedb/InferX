#include "inferx/ops/attention/capability.h"

#include <algorithm>
#include <string>
#include <variant>

namespace inferx::attn {
namespace {

/// The head dimension a capability's head_dims list is matched against.
/// 0 means "no meaningful head dim" (custom mechanisms): the check passes.
int64_t HeadDimOf(const AttentionSpec& spec) {
  return std::visit(
      [](const auto& p) -> int64_t {
        using T = std::decay_t<decltype(p)>;
        if constexpr (std::is_same_v<T, DenseParams>) {
          return p.head_dim;
        } else if constexpr (std::is_same_v<T, MlaParams>) {
          // The width a kernel specializes on is the full query-side key
          // width: nope + rope.
          return int64_t{p.qk_nope_dim} + p.qk_rope_dim;
        } else if constexpr (std::is_same_v<T, LinearParams>) {
          return p.k_dim;
        } else if constexpr (std::is_same_v<T, SparseParams>) {
          return p.base.head_dim;
        } else {
          return 0;
        }
      },
      spec.params);
}

constexpr int GraphRank(GraphSupport support) {
  return static_cast<int>(support);
}

Verdict Rejected(Reject why, std::string detail) {
  return Verdict{why, std::move(detail)};
}

std::string ToText(int64_t value) { return std::to_string(value); }

}  // namespace

Verdict CheckCapability(const Capability& capability, const AttentionSpec& spec, const Env& env) {
  // 1. Mechanism.
  const std::uint32_t mechanism = spec.MechanismId();
  if (std::find(capability.mechanisms.begin(), capability.mechanisms.end(), mechanism) ==
      capability.mechanisms.end()) {
    return Rejected(Reject::kMechanism,
                  "mechanism " + ToText(mechanism) + " not handled by " +
                      std::string(capability.name));
  }

  // 2. Role.
  if (!capability.roles.Has(spec.role)) {
    return Rejected(Reject::kRole, "role not supported by " + std::string(capability.name));
  }

  // 3. Mask.
  if (!capability.masks.Has(spec.mask.kind)) {
    return Rejected(Reject::kMask, "mask kind not supported by " + std::string(capability.name));
  }

  // 4. Features: every feature the layer uses must be supported.
  for (int f = 0; f < static_cast<int>(Feature::kCount); ++f) {
    const Feature feature = static_cast<Feature>(f);
    if (spec.features.Has(feature) && !capability.features.Has(feature)) {
      return Rejected(Reject::kFeature,
                    "feature not supported by " + std::string(capability.name));
    }
  }

  // 5/6. Dtypes.
  if (!capability.q_dtypes.Has(spec.q_dtype)) {
    return Rejected(Reject::kQDtype, "q dtype not supported by " + std::string(capability.name));
  }
  if (!capability.kv_dtypes.Has(spec.kv_dtype)) {
    return Rejected(Reject::kKvDtype,
                  "kv dtype not supported by " + std::string(capability.name));
  }

  // 7. Head dimension (empty list = any; 0 dim = nothing to check).
  const int64_t head_dim = HeadDimOf(spec);
  if (!capability.head_dims.empty() && head_dim != 0 &&
      std::find(capability.head_dims.begin(), capability.head_dims.end(),
                static_cast<std::int16_t>(head_dim)) == capability.head_dims.end()) {
    return Rejected(Reject::kHeadDim,
                  "head_dim " + ToText(head_dim) + " not in " + std::string(capability.name) +
                      "'s supported set");
  }

  // 8. Layout: the family declares what it writes; the backend must read it.
  if (!capability.layouts.Has(spec.kv_layout)) {
    return Rejected(Reject::kLayout,
                  "kv layout not readable by " + std::string(capability.name));
  }

  // 9. Block size against the resolved plan value (0 = not planned yet).
  if (env.block_size > 0) {
    if (capability.block.multiple_of > 1 && env.block_size % capability.block.multiple_of != 0) {
      return Rejected(Reject::kBlockSize,
                    "block size " + ToText(env.block_size) + " is not a multiple of " +
                        ToText(capability.block.multiple_of));
    }
    if (capability.block.max_tokens > 0 && env.block_size > capability.block.max_tokens) {
      return Rejected(Reject::kBlockSize, "block size " + ToText(env.block_size) +
                                            " exceeds the maximum of " +
                                            ToText(capability.block.max_tokens));
    }
  }

  // 10. Architecture (sm == 0 = unknown: skip).
  if (env.sm != 0) {
    if (env.sm < capability.min_sm ||
        (capability.max_sm > 0 && env.sm > capability.max_sm)) {
      return Rejected(Reject::kArch, "sm " + ToText(env.sm) + " outside [" +
                                       ToText(capability.min_sm) + ", " +
                                       ToText(capability.max_sm) + ")");
    }
  }

  // 11. Graph support: the runner's requirement must be met.
  if (GraphRank(capability.graph) < GraphRank(env.min_graph_support)) {
    return Rejected(Reject::kGraph,
                  "graph support below the required minimum for " +
                      std::string(capability.name));
  }

  // 12. GQA divisibility (dense-family mechanisms).
  const bool dense_family = std::holds_alternative<DenseParams>(spec.params) ||
                            std::holds_alternative<SparseParams>(spec.params);
  if (dense_family) {
    const DenseParams& dense = std::holds_alternative<DenseParams>(spec.params)
                                   ? std::get<DenseParams>(spec.params)
                                   : std::get<SparseParams>(spec.params).base;
    if (dense.num_q_heads % dense.num_kv_heads != 0) {
      return Rejected(Reject::kGqa, "num_q_heads " + ToText(dense.num_q_heads) +
                                      " not divisible by num_kv_heads " +
                                      ToText(dense.num_kv_heads));
    }
  }

  return Verdict{};
}

}  // namespace inferx::attn
