# inferx_core -- element types, tensors, status, and the memory layer.
#
# Everything here builds with the host compiler. Files that use the CUDA runtime
# API include <cuda_runtime_api.h>, which is plain C; only files containing
# __global__ kernels need nvcc, and there are none at M0.

add_library(inferx_core
  cpu/device_runtime.cc
  allocator.cc
  arena.cc
  caching_allocator.cc
  device_buffer.cc
  device_runtime.cc
  shape.cc
  storage.cc
  tensor.cc
  tensor_io.cc
)

if(INFERX_ENABLE_CUDA)
  target_sources(inferx_core PRIVATE cuda/device_runtime.cc)
  target_compile_definitions(inferx_core PRIVATE INFERX_WITH_CUDA=1)
  target_link_libraries(inferx_core PRIVATE CUDA::cudart)
else()
  target_sources(inferx_core PRIVATE accelerator_runtime_stub.cc)
endif()
add_library(inferx::core ALIAS inferx_core)
target_compile_features(inferx_core PUBLIC cxx_std_23)

target_include_directories(inferx_core PUBLIC
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>)

target_link_libraries(inferx_core
  PUBLIC
    absl::status
    absl::statusor
    absl::flat_hash_map
    absl::inlined_vector
    absl::span
    absl::strings
  PRIVATE
    mimalloc-static
)

target_compile_options(inferx_core PRIVATE -Wall -Wextra -Wpedantic)
