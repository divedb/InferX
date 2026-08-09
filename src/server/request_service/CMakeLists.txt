add_library(inferx_managed_request_service managed_request_service.cc)
add_library(inferx::managed_request_service ALIAS inferx_managed_request_service)
target_include_directories(inferx_managed_request_service PUBLIC
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>)
target_link_libraries(inferx_managed_request_service PUBLIC
  inferx::admission
  inferx::request_runtime
  inferx::scheduler_client
  inferx::support
  inferx::flags)
