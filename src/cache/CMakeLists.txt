# inferx_cache -- the paged KV block pool: host-side block bookkeeping for the
# scheduler over device-side KV storage.
inferx_add_library(
  NAME cache
  SRCS
    kv_block_pool.cc
    recurrent_state_pool.cc
  DEPS inferx::core
)
