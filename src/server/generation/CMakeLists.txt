add_library(inferx_generation generation.cc)
add_library(inferx::generation ALIAS inferx_generation)
target_include_directories(inferx_generation PUBLIC
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>)
target_link_libraries(inferx_generation PUBLIC
  inferx::core
  inferx::scheduler
  inferx::tokenizer
  inferx::folly_coro)
