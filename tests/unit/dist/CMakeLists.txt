# Collective contracts, single-rank copies, and concurrent loopback ranks.
inferx_add_test(
  NAME comm_test
  SRCS comm_test.cc
  DEPS inferx::dist
  TIMEOUT 30
  LABELS unit
  GPU
)
