add_library(inferx_scheduler_client INTERFACE)
add_library(inferx::scheduler_client ALIAS inferx_scheduler_client)
target_include_directories(inferx_scheduler_client INTERFACE
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>)
target_link_libraries(inferx_scheduler_client INTERFACE inferx::request_runtime
  inferx::folly_coro inferx::flags)
