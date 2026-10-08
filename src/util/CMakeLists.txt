# inferx_util -- typed, error-reporting views over HuggingFace JSON configs.
#
# Depends only on core status aliases and nlohmann/json so it stays buildable
# while the model-family modules are in flux.
inferx_add_library(
  NAME util
  SRCS
    config_view.cc
    hf_config.cc
  INCLUDE_DIRS ${PROJECT_SOURCE_DIR}/include
  DEPS
    inferx::core
    nlohmann_json::nlohmann_json
  PRIVATE_DEPS
    absl::status
    absl::statusor
    absl::strings
)
