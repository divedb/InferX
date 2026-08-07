add_library(inferx_engine_client ../engine_adapters.cc)
add_library(inferx::engine_client ALIAS inferx_engine_client)
target_include_directories(inferx_engine_client
  PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
  PRIVATE ${PROJECT_SOURCE_DIR}/src/server)
target_link_libraries(inferx_engine_client PUBLIC
  inferx::core
  inferx::scheduler
  inferx::scheduler_client
  inferx::server_observability
  inferx::flags)
