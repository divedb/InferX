# inferx_config -- device flags: parsing, validation, and resolution.
inferx_add_library(
  NAME config
  SRCS
    device_config.cc
  DEPS inferx::core
  PRIVATE_DEPS absl::log absl::strings
)
