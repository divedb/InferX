# Regression suites: executable records of investigations in docs/bugs.md.
# Each test pins a failure mode the project has actually hit -- greedy
# argmax tie semantics versus vLLM -- so a future change that regresses
# them fails here by name.
inferx_add_test(
  NAME argmax_tie_test
  SRCS argmax_tie_test.cc
  DEPS inferx::ops
  LABELS regression
  GPU
)
