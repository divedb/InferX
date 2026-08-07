add_library(inferx_streaming event_buffer.cc event_router.cc
  backpressure_controller.cc)
add_library(inferx::streaming ALIAS inferx_streaming)
target_include_directories(inferx_streaming PUBLIC
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>)
target_link_libraries(inferx_streaming PUBLIC inferx::scheduler_client
  inferx::core inferx::folly_coro inferx::flags)
