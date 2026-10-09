inferx_add_test(
  NAME scheduler_output_test
  SRCS scheduler_output_test.cc
  DEPS inferx::engine
  LABELS unit
)

inferx_add_test(
  NAME scheduler_test
  SRCS scheduler_test.cc
  DEPS inferx::engine
  LABELS unit
)
