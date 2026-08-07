set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
add_subdirectory(${PROJECT_SOURCE_DIR}/third_party/googletest
                 ${CMAKE_BINARY_DIR}/third_party/googletest
                 EXCLUDE_FROM_ALL)

add_executable(inferx_transport_test
  unit/transport/transport_test.cc)
target_link_libraries(inferx_transport_test PRIVATE
  inferx::http_transport
  GTest::gtest_main)

include(GoogleTest)
gtest_discover_tests(inferx_transport_test)
