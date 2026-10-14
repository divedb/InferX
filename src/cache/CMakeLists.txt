# inferx_cache -- logical request allocations and rank-local state storage.
inferx_add_library(
  NAME cache
  SRCS
    cache_manager.cc
    kv_block_pool.cc
    prefix_cache.cc
    recurrent_state_pool.cc
    sha256.cc
  DEPS
    inferx::core
    absl::flat_hash_map
    absl::span
)
