# inferx_sha256 -- self-contained SHA-256 (FIPS 180-4) digests. The prefix
# index and hash chain live in kv_block_pool and cache_manager.
inferx_add_library(
  NAME sha256
  SRCS
    sha256.cc
  INCLUDE_DIRS ${PROJECT_SOURCE_DIR}/include
)

# inferx_cache_spec -- declarative cache requirements and the group planner.
inferx_add_library(
  NAME cache_spec
  SRCS
    cache_spec.cc
  DEPS
    inferx::core
    absl::span
)

# inferx_kv_block_pool -- token-wise block storage with the per-block ledger
# (kvcache.md §6.2): refcounts, LRU eviction, absorbed prefix index.
inferx_add_library(
  NAME kv_block_pool
  SRCS
    kv_block_pool.cc
  DEPS
    inferx::cache_spec
    absl::flat_hash_map
    absl::span
)

# inferx_recurrent_state_pool -- fixed-slot per-sequence state storage for
# the recurrent group (kvcache.md §6.10).
inferx_add_library(
  NAME recurrent_state_pool
  SRCS
    recurrent_state_pool.cc
  DEPS
    inferx::core
    inferx::cache_spec
)

# inferx_cache_group_manager -- per-group lifecycle policies over the pools
# (kvcache.md §6.4): full attention, sliding window, chunked local,
# cross attention, recurrent.
inferx_add_library(
  NAME cache_group_manager
  SRCS
    cache_group_manager.cc
  DEPS
    inferx::kv_block_pool
    inferx::recurrent_state_pool
)

# inferx_cache_manager -- request-level facade over the cache groups:
# per-request state, hash chain, prefix adoption, atomic multi-group
# Reserve/Fork.
inferx_add_library(
  NAME cache_manager
  SRCS
    cache_manager.cc
  DEPS
    inferx::cache_group_manager
    inferx::sha256
)
