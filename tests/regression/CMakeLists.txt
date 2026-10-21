# Regression suites: executable records of investigations in docs/bugs.md.
# Each test pins a failure mode the project has actually hit -- greedy
# argmax tie semantics versus vLLM, and the cross-stack parity decision
# points -- so a future change that regresses them fails here by name.
inferx_add_test(
  NAME argmax_tie_test
  SRCS argmax_tie_test.cc
  DEPS inferx::ops
  LABELS regression
  GPU
)

inferx_add_test(
  NAME vllm_parity_probe_test
  SRCS vllm_parity_probe_test.cc
  DEPS inferx::engine inferx::prompt_service
  WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
  TIMEOUT 300
  LABELS regression
  GPU
)
