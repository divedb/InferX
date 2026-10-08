# inferx_cache -- logical request allocations and rank-local state storage.
inferx_add_library(
  NAME cache
  SRCS
    cache_manager.cc
    kv_block_pool.cc
    recurrent_state_pool.cc
  DEPS inferx::core
)
