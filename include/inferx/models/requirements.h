/// \file
/// \brief Persistent per-layer state declared by models and allocated by the
/// runtime. Layers describe *what* they need; the pool implementations own
/// the storage and lifecycle (reset, release, reuse).

#ifndef INFERX_MODELS_REQUIREMENTS_H_
#define INFERX_MODELS_REQUIREMENTS_H_

#include <cstdint>
#include <variant>
#include <vector>

#include "inferx/cache/kv_block_pool.h"

namespace inferx {

/// The runner-owned recurrent pool is defined in cache/recurrent_state_pool.h.
class RecurrentStatePool;

namespace models {

/// \brief Paged KV cache requirements declared by one attention layer.
struct PagedKvStateSpec {
  KvLayout layout;  ///< Per-token geometry the layer caches.
};

/// \brief Recurrent (linear-attention) state declared by one mixer layer.
struct RecurrentStateSpec {
  int64_t key_heads = 0;
  int64_t value_heads = 0;
  int64_t key_dim = 0;
  int64_t value_dim = 0;
  int64_t conv_kernel_size = 0;
};

/// \brief Persistent state one layer needs between steps.
using LayerStateSpec = std::variant<PagedKvStateSpec, RecurrentStateSpec>;

/// \brief Handle to one layer's region of the runner-owned paged KV pool.
struct PagedKvState {
  int64_t pool_layer = -1;  ///< Layer index into the pool.
};

/// \brief Handle to one layer's region of the runner-owned recurrent pool.
struct RecurrentState {
  int64_t pool_layer = -1;  ///< Layer index into the pool.
};

/// \brief Per-layer execution state owned by the runner and handed to
///        Forward each step.
struct ModelState {
  /// Pool backing every PagedKvState layer.
  const KvBlockPool* paged_kv = nullptr;

  /// Pool backing every RecurrentState layer; null when the model has none.
  const RecurrentStatePool* recurrent = nullptr;

  /// One entry per decoder layer, matching ModelRequirements order.
  std::vector<std::variant<PagedKvState, RecurrentState>> layers;
};

/// \brief The persistent state a model needs the runtime to allocate and
///        bind before execution. One entry per decoder layer, in execution
///        order; layer `i` of the model receives `layers[i]`.
struct ModelRequirements {
  std::vector<LayerStateSpec> layers;
};

}  // namespace inferx::models
}  // namespace inferx

#endif  // INFERX_MODELS_REQUIREMENTS_H_
