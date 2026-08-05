include(GoogleTest)

# Every test here must run to completion on a machine with no GPU. Tests that
# need a device gate themselves on inferx::CudaAvailable() and skip.
function(inferx_add_test name)
  add_executable(${name} ${ARGN})
  target_link_libraries(${name} PRIVATE
    inferx::core
    inferx::common
    inferx::model
    inferx::scheduler
    inferx::tokenizer
    inferx::api
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
    inferx::common
    inferx::model
    inferx::scheduler
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
inferx_add_test(config_test            unit/config_test.cc)
inferx_add_test(scheduler_test         unit/scheduler_test.cc)
inferx_add_test(prefix_cache_test      unit/prefix_cache_test.cc)
inferx_add_test(checkpoint_conformance_test unit/checkpoint_conformance_test.cc)
inferx_add_test(tokenizer_conformance_test unit/tokenizer_conformance_test.cc)
inferx_add_test(openai_api_test        unit/openai_api_test.cc)
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
  inferx_add_kernel_test(quantize_int4_test kernel/quantize_int4_test.cc)
  inferx_add_kernel_test(flashinfer_fp8_kv_test kernel/flashinfer_fp8_kv_test.cc)
  inferx_add_kernel_test(layers_test       kernel/layers_test.cc)
  inferx_add_kernel_test(moe_test         kernel/moe_test.cc)
  inferx_add_kernel_test(mla_test         kernel/mla_test.cc)
  inferx_add_kernel_test(mxfp4_test       kernel/mxfp4_test.cc)
  inferx_add_kernel_test(gpt_oss_test     kernel/gpt_oss_test.cc)
  inferx_add_kernel_test(qwen2_forward_test kernel/qwen2_forward_test.cc)
  inferx_add_kernel_test(qwen2_reference_test kernel/qwen2_reference_test.cc)
  inferx_add_kernel_test(paged_attention_test kernel/paged_attention_test.cc)
  inferx_add_kernel_test(qwen2_paged_test  kernel/qwen2_paged_test.cc)
  inferx_add_kernel_test(engine_test       kernel/engine_test.cc)
  if(INFERX_HAVE_FLASHINFER)
    inferx_add_kernel_test(flashinfer_test  kernel/flashinfer_test.cc)
  endif()

  # The server test drives a real socket, so it links the server and the HTTP
  # header on top of what a kernel test normally needs.
  if(INFERX_HAVE_HTTPLIB)
    inferx_add_kernel_test(server_test      kernel/server_test.cc)
    target_link_libraries(server_test PRIVATE
      inferx::server
      inferx::api
      inferx::cpp_httplib)

    # Its own binary, so the graphed and ungraphed engines it builds never
    # share a process with server_test's.
    inferx_add_kernel_test(server_graphs_test kernel/server_graphs_test.cc)
    target_link_libraries(server_graphs_test PRIVATE inferx::server)
  endif()
endif()
