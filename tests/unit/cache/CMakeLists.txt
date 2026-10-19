inferx_add_test(
  NAME cache_spec_test
  SRCS cache_spec_test.cc
  DEPS inferx::cache_spec
  LABELS unit
)
inferx_add_test(
  NAME kv_block_pool_test
  SRCS kv_block_pool_test.cc
  DEPS inferx::kv_block_pool
  LABELS unit
)
inferx_add_test(
  NAME cache_group_manager_test
  SRCS cache_group_manager_test.cc
  DEPS inferx::cache_group_manager
  LABELS unit
)
inferx_add_test(
  NAME cache_manager_test
  SRCS cache_manager_test.cc
  DEPS inferx::cache_manager
  LABELS unit
)
inferx_add_test(
  NAME sha256_test
  SRCS sha256_test.cc
  DEPS inferx::sha256
  LABELS unit
)
