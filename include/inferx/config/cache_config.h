/// \file
/// \brief KV cache pool sizing (vLLM CacheConfig analogue).

#ifndef INFERX_CACHE_CACHE_CONFIG_H_
#define INFERX_CACHE_CACHE_CONFIG_H_

#include <cstdint>

namespace inferx {

/// \brief Sizes the paged KV cache pool the runner allocates and the
///        scheduler allocates blocks from.
struct CacheConfig {
  /// \brief Total KV blocks to allocate across all layers.
  std::int64_t num_kv_blocks = 2048;
  /// \brief Tokens per KV block.
  std::int64_t block_size = 16;
  /// \brief Exact KV cache pool size in bytes; 0 derives the pool from
  ///        num_kv_blocks instead.
  std::int64_t kv_cache_memory_bytes = 0;
};

}  // namespace inferx

#endif  // INFERX_CACHE_CACHE_CONFIG_H_
