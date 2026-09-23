/// \file
/// \brief Persistent per-layer state contract between models and the runner.

#ifndef INFERX_MODELS_STATE_H_
#define INFERX_MODELS_STATE_H_

#include <cstdint>
#include <variant>
#include <vector>

#include "inferx/cache/kv_block_pool.h"

namespace inferx {

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

/// \brief Handle to one layer's recurrent state; device buffers pending.
struct RecurrentState {};

/// \brief Per-layer execution state owned by the ModelRunner and handed to
///        Model::Forward each step.
struct ModelState {
  /// Pool backing every PagedKvState layer.
  const KvBlockPool* paged_kv = nullptr;
  /// One entry per decoder layer, matching StateRequirements order.
  std::vector<std::variant<PagedKvState, RecurrentState>> layers;
};

}  // namespace inferx

#endif  // INFERX_MODELS_STATE_H_
