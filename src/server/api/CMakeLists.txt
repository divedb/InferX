# Host-only protocol policy shared by transport handlers.
add_library(inferx_server_protocol error_mapping.cc)
add_library(inferx::server_protocol ALIAS inferx_server_protocol)

target_include_directories(inferx_server_protocol PUBLIC
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>)
target_link_libraries(inferx_server_protocol PUBLIC
  inferx::api
  inferx::support
  inferx::flags)
