#ifndef INFERX_MODELS_REQUIREMENTS_H_
#define INFERX_MODELS_REQUIREMENTS_H_

#include <cstdint>
#include <vector>

#include "inferx/cache/kv_block_pool.h"

namespace inferx {
namespace models {

/// \brief Fixed geometry of one recurrent (linear-attention) layer's state.
///
/// Layers such as Qwen3-Next's gated delta rule carry per-sequence state
/// that cannot live in the paged KV pool: a delta-rule matrix
/// [value_heads, key_dim, value_dim] and a causal-conv input ring. The pool
/// only stores the geometry; kernels interpret the regions.
struct RecurrentStateSpec {
  int64_t key_heads = 0;        ///< Key heads of the mixer (divides value_heads).
  int64_t value_heads = 0;      ///< Value heads of the mixer.
  int64_t key_dim = 0;          ///< Per-head key dimension.
  int64_t value_dim = 0;        ///< Per-head value dimension.
  int64_t conv_kernel_size = 0; ///< Causal-conv kernel width (>= 1).
};

/// \brief Per-layer state a model family needs the runtime to keep.
///
/// The model layer describes state; it never allocates it. The runner sizes
/// the shared pools from these specs at startup, so a model that needs no
/// extra state simply contributes nothing here.
struct StateRequirements {
  /// Paged KV geometry, one entry per decoder layer in model order.
  std::vector<KvLayout> paged_kv;

  /// Recurrent mixer state, one entry per recurrent layer in model order.
  std::vector<RecurrentStateSpec> recurrent;
};

}  // namespace models
}  // namespace inferx

#endif  // INFERX_MODELS_REQUIREMENTS_H_
