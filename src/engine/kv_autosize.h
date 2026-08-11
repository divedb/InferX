#pragma once

#include <cstdint>

#include "inferx/core/status.h"

namespace inferx::engine {

/// Post-attach allocations the budget cannot see yet -- FlashInfer workspace,
/// graph pools, allocator slack. Subtracted from the utilization budget so
/// startup does not OOM moments after the KV pool lands.
inline constexpr int64_t kKvAutosizeHeadroomBytes = 1ll << 30;  // 1 GiB

/// How one rank's KV pool should be sized. Mirrors vLLM's precedence:
/// an explicit block count pins the pool, an explicit byte budget is next,
/// and otherwise the pool takes whatever `gpu_memory_utilization` leaves
/// after what is already allocated (weights, activations) plus headroom.
struct KvSizingSpec {
  int64_t explicit_blocks = 0;         // --num-gpu-blocks-override; >0 wins
  int64_t explicit_bytes = 0;          // --kv-cache-memory-bytes; >0 next
  double gpu_memory_utilization = 0.92;
  int64_t block_bytes = 0;             // this rank's bytes per block
  int64_t min_blocks = 1;              // one full max-model-len sequence
  int64_t headroom_bytes = kKvAutosizeHeadroomBytes;
};

/// The pure arithmetic, separated so it unit-tests without a device.
/// `free_bytes`/`total_bytes` are cudaMemGetInfo's view of the current device
/// *after* weights are loaded, so `total - free` is what the model already
/// occupies and the budget is `util * total - (total - free) - headroom`.
StatusOr<int64_t> ResolveKvBlocks(const KvSizingSpec& spec, int64_t free_bytes,
                                  int64_t total_bytes);

/// Measures the current device and resolves, logging the arithmetic the way
/// vLLM does at startup. Call after weights load, before AttachKvCache.
StatusOr<int64_t> AutosizeKvBlocksOnCurrentDevice(const KvSizingSpec& spec);

}  // namespace inferx::engine
