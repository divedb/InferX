#ifndef INFERX_CACHE_CACHE_CONFIG_H_
#define INFERX_CACHE_CACHE_CONFIG_H_

#include <cstdint>

namespace inferx {

/// \brief Sizes the paged KV cache pool the runner allocates and the
///        scheduler allocates blocks from.
struct CacheConfig {
  /// Total KV blocks to allocate across all layers.
  /// EXAMPLE: --num-kv-blocks 4096
  std::int64_t num_kv_blocks = 2048;

  /// Tokens per KV block.
  /// EXAMPLE: --block-size 16
  std::int64_t block_size = 16;

  /// Exact KV cache pool size in bytes; 0 derives the pool from num_kv_blocks
  /// instead.
  /// EXAMPLE: --kv-cache-memory-bytes 8589934592
  std::int64_t kv_cache_memory_bytes = 0;

  /// Fraction of device memory the engine may use for weights + KV; when
  /// positive the runner profiles one max-size step after loading and
  /// gives the leftover budget to the KV pool (vLLM-style sizing, minimum
  /// taken across the tensor-parallel group). 0 sizes from the fields
  /// above, which co-located ranks on one device should prefer.
  /// EXAMPLE: --gpu-memory-utilization 0.9
  float gpu_memory_utilization = 0.0f;

  /// When true, the scheduler will cache the prefix KV blocks of each
  /// sequence and re-use them across steps. When false, the scheduler will
  /// always re-fill the KV cache from the input tokens. This is only relevant
  /// for multi-step sequences (prefill + decode) and is ignored for pure
  /// decode sequences. The default is true, which is the most common use case.
  /// EXAMPLE: --enable-prefix-caching true
  bool enable_prefix_caching = true;
};

}  // namespace inferx

#endif  // INFERX_CACHE_CACHE_CONFIG_H_
