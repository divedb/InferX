inferx_add_test(
  NAME cache_manager_test
  SRCS cache_manager_test.cc
  DEPS inferx::cache
  LABELS unit
)
inferx_add_test(
  NAME cache_manager_prefix_test
  SRCS cache_manager_prefix_test.cc
  DEPS inferx::cache
  LABELS unit
)
inferx_add_test(
  NAME prefix_cache_test
  SRCS prefix_cache_test.cc
  DEPS inferx::cache
  LABELS unit
)
