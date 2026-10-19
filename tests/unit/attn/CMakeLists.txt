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
inferx_add_test(
  NAME attention_selector_test
  SRCS attention_selector_test.cc
  DEPS inferx::attn
  LABELS unit
)
inferx_add_test(
  NAME attention_runtime_test
  SRCS attention_runtime_test.cc
  DEPS inferx::attn
  LABELS unit
)
inferx_add_test(
  NAME attention_flash_gpu_test
  SRCS attention_flash_gpu_test.cc
  DEPS inferx::attn
  LABELS unit
  GPU
)
