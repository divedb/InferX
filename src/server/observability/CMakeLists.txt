add_library(inferx_server_observability metrics_snapshot.cc)
add_library(inferx::server_observability ALIAS inferx_server_observability)
target_include_directories(inferx_server_observability PUBLIC
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>)
target_link_libraries(inferx_server_observability PUBLIC
  inferx::observe
  inferx::flags)
