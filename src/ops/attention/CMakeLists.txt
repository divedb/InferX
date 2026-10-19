# inferx_attn -- the unified attention architecture, host side
# (docs/attention.md plan steps 1-2): spec vocabulary, the cache-spec
# bridge, capability matching, and kernel-facing cache views. Kernels,
# builders, and backends land in later plan steps.
inferx_add_library(
  NAME attn
  SRCS
    cache_spec_bridge.cc
    capability.cc
    cache_view.cc
  DEPS
    inferx::core
    inferx::cache_spec
    inferx::kv_block_pool
    inferx::recurrent_state_pool
)
