# Model-layer unit tests. The safetensors suites are restored from the
# loading infrastructure's own history (view = pure parser over a byte
# buffer, reader = boost mmap over a file); the checkpoint suite covers the
# directory-level reader; the config and registry suites are CPU-only; the
# forward gate runs a tiny synthetic model on the GPU and compares against
# the FP32 reference implemented in the test.
inferx_add_test(
  NAME safe_tensors_view_test
  SRCS safe_tensors_view_test.cc
  DEPS inferx::models
  LABELS unit
)

inferx_add_test(
  NAME safe_tensors_reader_test
  SRCS safe_tensors_reader_test.cc
  DEPS inferx::models
  LABELS unit
)

inferx_add_test(
  NAME scratch_arena_test
  SRCS scratch_arena_test.cc
  DEPS inferx::models
  LABELS unit
  GPU
)

inferx_add_test(
  NAME checkpoint_reader_test
  SRCS checkpoint_reader_test.cc
  DEPS inferx::models
  LABELS unit
)

inferx_add_test(
  NAME model_config_test
  SRCS config_test.cc
  DEPS inferx::models
  LABELS unit
)

inferx_add_test(
  NAME model_registry_test
  SRCS registry_test.cc
  DEPS inferx::models
  LABELS unit
)

inferx_add_test(
  NAME qwen3_forward_test
  SRCS qwen3_forward_test.cc
  DEPS inferx::models inferx::comm
  LABELS unit
  GPU
)
