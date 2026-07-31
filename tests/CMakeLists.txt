include(GoogleTest)

# Every test here must run to completion on a machine with no GPU. Tests that
# need a device gate themselves on inferx::CudaAvailable() and skip.
function(inferx_add_test name)
  add_executable(${name} ${ARGN})
  target_link_libraries(${name} PRIVATE
    inferx::core
    inferx::flags
    GTest::gtest_main
  )
  gtest_discover_tests(${name}
    PROPERTIES LABELS unit
    DISCOVERY_TIMEOUT 30
  )
endfunction()

inferx_add_test(dtype_test             unit/dtype_test.cc)
inferx_add_test(intrusive_ref_cnt_ptr_test unit/intrusive_ref_cnt_ptr_test.cc)
inferx_add_test(shape_test             unit/shape_test.cc)
inferx_add_test(tensor_view_test       unit/tensor_view_test.cc)
inferx_add_test(tensor_test            unit/tensor_test.cc)
inferx_add_test(arena_test             unit/arena_test.cc)
inferx_add_test(caching_allocator_test unit/caching_allocator_test.cc)
inferx_add_test(device_buffer_test     unit/device_buffer_test.cc)
