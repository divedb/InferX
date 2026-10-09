inferx_add_test(
  NAME tensor_test
  SRCS tensor_test.cc
  DEPS inferx::core
  LABELS unit
  GPU
)

inferx_add_test(
  NAME logging_test
  SRCS logging_test.cc
  DEPS inferx::core
  LABELS unit
)
