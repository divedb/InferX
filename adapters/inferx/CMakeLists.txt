add_library(inferx_kvc_adapter provider_memory.cc)
add_library(inferx::kvc_adapter ALIAS inferx_kvc_adapter)
target_link_libraries(inferx_kvc_adapter PUBLIC inferx::core kvc::runtime)
if(INFERX_BUILD_TESTS AND INFERX_BUILD_KVCACHE_PLUGINS)
  add_executable(inferx_kvc_adapter_test ${PROJECT_SOURCE_DIR}/tests/kvc/adapter_test.cc)
  target_link_libraries(inferx_kvc_adapter_test PRIVATE inferx::kvc_adapter)
  add_dependencies(inferx_kvc_adapter_test kvc_host)
  add_test(NAME kvc_adapter COMMAND inferx_kvc_adapter_test $<TARGET_FILE:kvc_host>)
  if(TARGET kvc_cuda)
    add_dependencies(inferx_kvc_adapter_test kvc_cuda)
    target_link_libraries(inferx_kvc_adapter_test PRIVATE CUDA::cudart)
    target_compile_definitions(inferx_kvc_adapter_test PRIVATE INFERX_TEST_CUDA=1)
    set_tests_properties(kvc_adapter PROPERTIES ENVIRONMENT
                         "KVC_ADAPTER_CUDA=$<TARGET_FILE:kvc_cuda>")
  endif()
  if(NOT MSVC)
    target_compile_options(inferx_kvc_adapter_test PRIVATE -Wall -Wextra -Wpedantic)
  endif()
endif()


if(EXISTS "/usr/lib/wsl/lib/libcuda.so.1")
  # WSL2 ships two libcuda copies; only the WSL one can see the GPU.
  set_tests_properties(kvc_adapter PROPERTIES ENVIRONMENT
                       "LD_LIBRARY_PATH=/usr/lib/wsl/lib")
endif()
