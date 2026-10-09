# JSON config parsing. Linked only against inferx::util so these stay
# buildable while the models/engine modules are mid-refactor.
inferx_add_test(
  NAME config_view_test
  SRCS config_view_test.cc
  DEPS inferx::util
  LABELS unit
)

inferx_add_test(
  NAME hf_config_test
  SRCS hf_config_test.cc
  DEPS inferx::util
  LABELS unit
)
