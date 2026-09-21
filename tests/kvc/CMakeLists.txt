# KV-cache provider tests. These build with INFERX_BUILD_ENGINE=OFF: the
# suite exercises the SDK, the bundled providers, and the loader, without
# any engine dependency. adapter_test.cc is wired from adapters/inferx.

find_package(Threads REQUIRED)

# The ABI headers must stay compilable as plain C11.
add_library(kvc_c_header_check STATIC header_check.c)
target_link_libraries(kvc_c_header_check PUBLIC kvc::abi)
set_target_properties(kvc_c_header_check PROPERTIES C_STANDARD 11 C_STANDARD_REQUIRED ON)

# Loader fixtures: an entry point with a bad ABI major and a library with
# no entry point at all.
add_library(kvc_bad_abi MODULE bad_abi.c)
target_link_libraries(kvc_bad_abi PRIVATE kvc::abi)
add_library(kvc_missing_entry MODULE missing_entry.c)
target_link_libraries(kvc_missing_entry PRIVATE kvc::abi)

add_executable(kvc_loader_test loader_test.cc)
target_link_libraries(kvc_loader_test PRIVATE kvc::runtime Threads::Threads)
add_test(NAME kvc_loader COMMAND kvc_loader_test $<TARGET_FILE:kvc_bad_abi>
         $<TARGET_FILE:kvc_missing_entry> $<TARGET_FILE:kvc_host>)

# A provider with delayed background transfers and failure injection.
add_library(kvc_async_fixture MODULE async_fixture.cc)
target_link_libraries(kvc_async_fixture PRIVATE kvc::abi Threads::Threads)
set_target_properties(kvc_async_fixture PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)

add_executable(kvc_provider_conformance provider_conformance.cc)
target_link_libraries(kvc_provider_conformance PRIVATE kvc::runtime Threads::Threads)
set_target_properties(kvc_provider_conformance PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
add_test(NAME kvc_provider_conformance COMMAND kvc_provider_conformance
         $<TARGET_FILE:kvc_host>)
add_test(NAME kvc_async_lifetime COMMAND kvc_provider_conformance
         $<TARGET_FILE:kvc_async_fixture> async)

if(TARGET kvc_cuda)
  find_package(CUDAToolkit REQUIRED)
  add_executable(kvc_cuda_provider_test cuda_provider_test.cc)
  target_link_libraries(kvc_cuda_provider_test PRIVATE kvc::runtime CUDA::cudart
                                                   Threads::Threads)
  set_target_properties(kvc_cuda_provider_test PROPERTIES CXX_STANDARD 23
                                                         CXX_STANDARD_REQUIRED ON)
  add_test(NAME kvc_cuda_provider COMMAND kvc_cuda_provider_test $<TARGET_FILE:kvc_cuda>)
  add_test(NAME kvc_cuda_conformance COMMAND kvc_provider_conformance
           $<TARGET_FILE:kvc_cuda> cuda)
endif()

foreach(kvc_test_target kvc_loader_test kvc_provider_conformance kvc_bad_abi
         kvc_missing_entry kvc_async_fixture)
  if(NOT MSVC)
    target_compile_options(${kvc_test_target} PRIVATE -Wall -Wextra -Wpedantic)
  endif()
endforeach()
if(TARGET kvc_cuda_provider_test AND NOT MSVC)
  target_compile_options(kvc_cuda_provider_test PRIVATE -Wall -Wextra -Wpedantic)
endif()


if(EXISTS "/usr/lib/wsl/lib/libcuda.so.1")
  # WSL2 ships two libcuda copies; only the WSL one can see the GPU.
  set_tests_properties(kvc_cuda_provider kvc_cuda_conformance
                       PROPERTIES ENVIRONMENT "LD_LIBRARY_PATH=/usr/lib/wsl/lib")
endif()
