add_library(inferx_server_middleware authentication.cc)
add_library(inferx::server_middleware ALIAS inferx_server_middleware)
target_include_directories(inferx_server_middleware PUBLIC
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>)
target_link_libraries(inferx_server_middleware PUBLIC
  inferx::auth
  inferx::http_transport
  inferx::flags)
