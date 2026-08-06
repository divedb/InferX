add_library(inferx_comm
  communicator.cc
)
add_library(inferx::comm ALIAS inferx_comm)

target_include_directories(inferx_comm PUBLIC
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>)

target_link_libraries(inferx_comm PUBLIC
  inferx::core
  inferx::flags
)
