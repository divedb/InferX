add_library(inferx_admission
  admission_controller.cc
  capacity_policy.cc)
add_library(inferx::admission ALIAS inferx_admission)
target_include_directories(inferx_admission PUBLIC
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>)
target_link_libraries(inferx_admission PUBLIC inferx::core inferx::folly_coro
  inferx::flags)
