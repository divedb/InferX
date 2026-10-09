inferx_add_test(
  NAME cli_args_test
  SRCS cli_args_test.cc
  DEPS inferx::cli
  LABELS unit
)

inferx_add_test(
  NAME console_help_test
  SRCS console_help_test.cc
  DEPS inferx::cli
  LABELS unit
)
