inferx_add_test(
  NAME parity_test
  SRCS parity_test.cc
  DEPS inferx::models
  WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
  LABELS integration
  GPU
)

inferx_add_test(
  NAME model_test
  SRCS model_test.cc
  DEPS inferx::engine
  WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
  LABELS integration
  GPU
)

inferx_add_test(
  NAME tensor_parallel_test
  SRCS tensor_parallel_test.cc
  DEPS inferx::engine
  WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
  TIMEOUT 120
  LABELS integration
  GPU
)
