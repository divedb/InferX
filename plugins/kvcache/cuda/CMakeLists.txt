include(GNUInstallDirs)
# CUDA tiered provider: GPU-resident storage with LRU offload to a host
# tier and promotion back on load. Links the CUDA runtime directly; the
# provider SDK itself stays runtime-agnostic.
find_package(CUDAToolkit REQUIRED)
add_library(kvc_cuda MODULE provider.cc transfer.cc tables.cc)
target_link_libraries(kvc_cuda PRIVATE kvc::abi CUDA::cudart)
set_target_properties(kvc_cuda PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
if(NOT MSVC)
  target_compile_options(kvc_cuda PRIVATE -Wall -Wextra -Wpedantic -fvisibility=hidden)
else()
  target_compile_options(kvc_cuda PRIVATE /w)
endif()
install(TARGETS kvc_cuda LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}/kvcache/providers)
