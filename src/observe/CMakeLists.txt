add_library(inferx_observe
  metrics.cc
)
add_library(inferx::observe ALIAS inferx_observe)

target_include_directories(inferx_observe PUBLIC
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>)

target_link_libraries(inferx_observe PUBLIC inferx::flags)
