# inferx_model -- checkpoint loading and model definitions.
#
# Host-only and GPU-free: this layer reads files and produces host tensors. It
# builds in a --host-only configuration on purpose, so that loader conformance
# tests run without a device, the same way the scheduler will (§3.1).

add_library(inferx_model
  config.cc
  safetensors.cc
)
add_library(inferx::model ALIAS inferx_model)

target_link_libraries(inferx_model
  PUBLIC
    inferx::core
    inferx::common
    inferx::flags
)
