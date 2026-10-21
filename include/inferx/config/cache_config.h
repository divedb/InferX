#ifndef INFERX_CACHE_CACHE_CONFIG_H_
#define INFERX_CACHE_CACHE_CONFIG_H_

#include <cstdint>
#include <string>

#include "inferx/core/status.h"

namespace inferx {

/// \brief Sizes the paged KV cache pool the runner allocates and the
///        scheduler allocates blocks from.
///
/// Pool size is resolved in this order (first match wins):
///   1. num_gpu_blocks_override > 0: use exactly that many blocks.
///   2. kv_cache_memory_bytes > 0: pool size per device in bytes.
///   3. gpu_memory_utilization: profile one max-size step after loading and
///      give the leftover budget to the KV pool.
struct CacheConfig {
  /// Tokens per KV block. One block id spans all layers.
  /// EXAMPLE: --block-size 16
  std::int64_t block_size = 16;

  /// Fraction of device memory this engine instance may use for weights,
  /// activations and KV. Must be in (0, 1]. Minimum is taken across the
  /// tensor-parallel group.
  /// EXAMPLE: --gpu-memory-utilization 0.9
  float gpu_memory_utilization = 0.9f;

  /// KV pool size per device in bytes. 0 = unset. When set, it takes
  /// precedence over gpu_memory_utilization for KV sizing.
  /// EXAMPLE: --kv-cache-memory-bytes 8589934592
  std::int64_t kv_cache_memory_bytes = 0;

  /// Forces the total KV block count (vLLM: --num-gpu-blocks-override).
  /// 0 = unset. Replaces `num_kv_blocks`; mainly for tests and debugging.
  /// EXAMPLE: --num-gpu-blocks-override 4096
  std::int64_t num_gpu_blocks_override = 0;

  /// KV cache storage dtype: "auto" follows the model dtype; also "fp8",
  /// "fp8_e4m3", "fp8_e5m2".
  /// EXAMPLE: --kv-cache-dtype fp8
  std::string cache_dtype = "auto";

  /// Compute per-tensor KV scales at runtime for fp8 KV caches.
  /// EXAMPLE: --calculate-kv-scales
  bool calculate_kv_scales = false;

  /// Reuse KV blocks of shared prompt prefixes across requests. Only affects
  /// prefill; pure decode steps ignore it.
  /// EXAMPLE: --enable-prefix-caching / --no-enable-prefix-caching
  bool enable_prefix_caching = true;

  /// Hash used for prefix-cache block keys: "sha256" or "sha256_cbor".
  /// vLLM's "builtin" (Python hash()) has no C++ equivalent.
  /// EXAMPLE: --prefix-caching-hash-algo sha256
  std::string prefix_caching_hash_algo = "sha256";

  /// Sliding-window size in tokens; 0 = none. Normally derived from the
  /// model config rather than set by the user.
  std::int64_t sliding_window = 0;

  Status Validate() const;
};

}  // namespace inferx

#endif  // INFERX_CACHE_CACHE_CONFIG_H_
