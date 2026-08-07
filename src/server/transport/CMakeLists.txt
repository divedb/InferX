# Host-only contracts shared by coroutine handlers and the Beast adapter.
# Beast remains the only HTTP representation and Folly remains the only
# cancellation/coroutine representation.

add_library(inferx_http_transport INTERFACE)
add_library(inferx::http_transport ALIAS inferx_http_transport)

target_include_directories(inferx_http_transport INTERFACE
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>)

target_link_libraries(inferx_http_transport INTERFACE
  inferx::core
  inferx::flags
  inferx::beast
  inferx::folly_coro)
