add_library(inferx_scheduler_client in_process_scheduler_client.cc)
add_library(inferx::scheduler_client ALIAS inferx_scheduler_client)
target_include_directories(inferx_scheduler_client PUBLIC
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>)
target_link_libraries(inferx_scheduler_client PUBLIC inferx::request_runtime
  inferx::folly_coro inferx::flags)
