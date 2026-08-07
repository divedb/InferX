# Host-only contracts shared by coroutine handlers and the Beast adapter.
# Beast remains the only HTTP representation and Folly remains the only
# cancellation/coroutine representation.

add_library(inferx_http_transport
  beast_folly_adapter.cc
  beast_listener.cc
  beast_session.cc
  io_runtime.cc
  routes.cc
  sse_writer.cc
)
add_library(inferx::http_transport ALIAS inferx_http_transport)

target_include_directories(inferx_http_transport PUBLIC
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>)

target_link_libraries(inferx_http_transport PUBLIC
  inferx::core
  inferx::flags
  inferx::beast
  inferx::folly_coro)
