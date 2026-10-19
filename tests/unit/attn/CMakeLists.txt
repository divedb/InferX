inferx_add_test(
  NAME attention_spec_test
  SRCS attention_spec_test.cc
  DEPS inferx::attn
  LABELS unit
)
inferx_add_test(
  NAME attention_cache_spec_bridge_test
  SRCS cache_spec_bridge_test.cc
  DEPS inferx::attn
  LABELS unit
)
inferx_add_test(
  NAME attention_capability_test
  SRCS capability_test.cc
  DEPS inferx::attn
  LABELS unit
)
inferx_add_test(
  NAME attention_cache_view_test
  SRCS cache_view_test.cc
  DEPS inferx::attn
  LABELS unit
)
