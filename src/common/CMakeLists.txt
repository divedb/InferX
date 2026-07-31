# inferx_common -- small utilities with no device or model dependencies.

add_library(inferx_common
  json.cc
)
add_library(inferx::common ALIAS inferx_common)

target_include_directories(inferx_common PUBLIC
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>)

target_link_libraries(inferx_common
  PUBLIC
    inferx::flags
    absl::status
    absl::statusor
    absl::strings
)
