# inferx_support -- small utilities with no device or model dependencies.

add_library(inferx_support
  file.cc
  json.cc
)
add_library(inferx::support ALIAS inferx_support)

target_include_directories(inferx_support PUBLIC
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>)

target_link_libraries(inferx_support
  PUBLIC
    inferx::flags
    absl::status
    absl::statusor
    absl::strings
)
