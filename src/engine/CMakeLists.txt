# inferx_engine -- device-backed model execution and scheduling orchestration.

add_library(inferx_engine
  engine.cc
  kv_autosize.cc
  qwen_runner.cc)
add_library(inferx::engine ALIAS inferx_engine)

target_include_directories(inferx_engine
  PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
  PRIVATE ${PROJECT_SOURCE_DIR}/src)

target_link_libraries(inferx_engine
  PUBLIC
    inferx::core
    inferx::model
    inferx::scheduler
    inferx::tokenizer
    inferx::folly_coro
    inferx::flags
  PRIVATE
    inferx::api
    inferx::generation
    inferx::observe
    inferx::support)

find_package(Threads REQUIRED)
target_link_libraries(inferx_engine PRIVATE Threads::Threads)

# Bridges the engine API to the host-only scheduler-client and metrics
# contracts consumed by the HTTP server.
add_library(inferx_engine_adapters engine_adapters.cc)
add_library(inferx::engine_adapters ALIAS inferx_engine_adapters)
target_include_directories(inferx_engine_adapters
  PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>)
target_link_libraries(inferx_engine_adapters PUBLIC
  inferx::engine
  inferx::scheduler_client
  inferx::server_observability
  inferx::flags)
