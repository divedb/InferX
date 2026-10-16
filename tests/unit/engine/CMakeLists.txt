inferx_add_test(
  NAME request_test
  SRCS request_test.cc
  DEPS inferx::engine
  LABELS unit
)

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
inferx_add_test(
  NAME scheduler_prefix_test
  SRCS scheduler_prefix_test.cc
  DEPS inferx::engine
  LABELS unit
)
inferx_add_test(
  NAME scheduler_preempt_test
  SRCS scheduler_preempt_test.cc
  DEPS inferx::engine
  LABELS unit
)
