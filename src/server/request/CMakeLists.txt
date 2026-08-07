add_library(inferx_request_runtime
  request_id.cc
  state_machine.cc)
add_library(inferx::request_runtime ALIAS inferx_request_runtime)

target_include_directories(inferx_request_runtime PUBLIC
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>)
target_link_libraries(inferx_request_runtime PUBLIC inferx::core inferx::folly_coro
  inferx::flags)
