include(GoogleTest)

# Every test here must run to completion on a machine with no GPU. Tests that
# need a device gate themselves on inferx::CudaAvailable() and skip.
function(inferx_add_test name)
  add_executable(${name} ${ARGN})
  target_link_libraries(${name} PRIVATE
    inferx::core
    inferx::common
    inferx::model
    inferx::flags
    GTest::gtest_main
    GTest::gmock
  )
  gtest_discover_tests(${name}
    PROPERTIES LABELS unit
    DISCOVERY_TIMEOUT 30
  )
endfunction()

# Kernel tests link the device layer and are labelled separately, so that
# `ctest -L unit` stays the no-GPU-required suite. Like every GPU test they
# skip at runtime when no device is present -- the label is about what they
# link, the skip is about what is plugged in.
function(inferx_add_kernel_test name)
  add_executable(${name} ${ARGN})
  target_link_libraries(${name} PRIVATE
    inferx::core
    inferx::kernels
    inferx::flags
    GTest::gtest_main
  )
  gtest_discover_tests(${name}
    PROPERTIES LABELS kernel
    DISCOVERY_TIMEOUT 30
  )
endfunction()

inferx_add_test(dtype_test             unit/dtype_test.cc)
inferx_add_test(json_test              unit/json_test.cc)
inferx_add_test(safetensors_test       unit/safetensors_test.cc)
inferx_add_test(checkpoint_conformance_test unit/checkpoint_conformance_test.cc)
inferx_add_test(intrusive_ref_cnt_ptr_test unit/intrusive_ref_cnt_ptr_test.cc)
inferx_add_test(shape_test             unit/shape_test.cc)
inferx_add_test(tensor_view_test       unit/tensor_view_test.cc)
inferx_add_test(tensor_test            unit/tensor_test.cc)
inferx_add_test(arena_test             unit/arena_test.cc)
inferx_add_test(caching_allocator_test unit/caching_allocator_test.cc)
inferx_add_test(device_buffer_test     unit/device_buffer_test.cc)

if(INFERX_CUDA_MEETS_FLOOR)
  inferx_add_kernel_test(kernel_smoke_test kernel/smoke_test.cc)
  inferx_add_kernel_test(gemm_test         kernel/gemm_test.cc)
  inferx_add_kernel_test(gemm_fp8_test     kernel/gemm_fp8_test.cc)
endif()
