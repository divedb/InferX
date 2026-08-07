set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
add_subdirectory(${PROJECT_SOURCE_DIR}/third_party/googletest
                 ${CMAKE_BINARY_DIR}/third_party/googletest
                 EXCLUDE_FROM_ALL)

add_executable(inferx_transport_test
  unit/transport/transport_test.cc)
target_link_libraries(inferx_transport_test PRIVATE
  inferx::http_transport
  inferx::server_protocol
  inferx::admission
  inferx::auth
  inferx::server_config
  inferx::model_registry
  inferx::tokenization_service
  inferx::request_runtime
  inferx::managed_request_service
  inferx::streaming
  inferx::server_middleware
  inferx::request_handlers
  GTest::gtest_main)

include(GoogleTest)
gtest_discover_tests(inferx_transport_test)

add_executable(inferx_remote_scheduler_client_test
  unit/scheduler_client/remote_scheduler_client_test.cc)
target_link_libraries(inferx_remote_scheduler_client_test PRIVATE
  inferx::scheduler_client
  GTest::gtest_main)
gtest_discover_tests(inferx_remote_scheduler_client_test)
