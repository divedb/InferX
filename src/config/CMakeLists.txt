# inferx_config -- engine-independent configuration: device flags, cache
# validation, generation settings, and the serving frontend's tunables
# (server_config).
inferx_add_library(
  NAME config
  SRCS
    device_config.cc
    cache_config.cc
    generation_config.cc
    merge_config.cc
    server_config.cc
  HDRS
    ${PROJECT_SOURCE_DIR}/include/inferx/config/generation_config.h
    ${PROJECT_SOURCE_DIR}/include/inferx/config/merge_config.h
    ${PROJECT_SOURCE_DIR}/include/inferx/config/server_config.h
  DEPS inferx::core inferx::util
  PRIVATE_DEPS absl::log absl::strings nlohmann_json::nlohmann_json
)
