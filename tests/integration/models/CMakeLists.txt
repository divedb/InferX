# Whole-model integration: the golden-trace gate runs the scheduler and
# runner over the repository's Qwen3-0.6B checkpoint; the tensor-parallel
# gate drives a LoopbackWorld rank group through ModelExecutor.
inferx_add_test(
  NAME model_test
  SRCS model_test.cc
  DEPS inferx::models inferx::engine
  WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
  LABELS integration
  GPU
)

inferx_add_test(
  NAME tensor_parallel_test
  SRCS tensor_parallel_test.cc
  DEPS inferx::engine inferx::dist
  WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
  TIMEOUT 300
  LABELS integration
  GPU
)

