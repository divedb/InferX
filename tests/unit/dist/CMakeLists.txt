# Collective contracts, single-rank copies, and concurrent loopback ranks.
inferx_add_test(
  NAME comm_test
  SRCS comm_test.cc
  DEPS inferx::dist
  TIMEOUT 30
  LABELS unit
  GPU
)
inferx_add_test(
  NAME topology_test
  SRCS topology_test.cc
  DEPS inferx::topology
  LABELS unit
)
inferx_add_test(
  NAME shard_test
  SRCS shard_test.cc
  DEPS inferx::shard
  LABELS unit
)
