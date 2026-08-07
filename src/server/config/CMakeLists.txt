add_library(inferx_server_config server_config.cc)
add_library(inferx::server_config ALIAS inferx_server_config)
target_include_directories(inferx_server_config PUBLIC
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>)
target_link_libraries(inferx_server_config PUBLIC inferx::core inferx::flags)
