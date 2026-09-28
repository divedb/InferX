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
};

}  // namespace inferx

#endif  // INFERX_CACHE_CACHE_CONFIG_H_
