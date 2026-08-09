# inferx_support -- small utilities with no device or model dependencies.

add_library(inferx_support
  crash.cc
  file.cc
  json.cc
  log.cc
  shutdown.cc
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
    absl::log
    absl::log_initialize
    absl::log_globals
    absl::log_sink_registry
    absl::time
    absl::failure_signal_handler
    absl::symbolize
)
