# inferx_core -- element types, tensors, status, and the memory layer.
#
# Everything here builds with the host compiler. Files that use the CUDA runtime
# API include <cuda_runtime_api.h>, which is plain C; only files containing
# __global__ kernels need nvcc, and there are none at M0.

add_library(inferx_core
  allocator.cc
  arena.cc
  caching_allocator.cc
  cuda_utils.cc
  device_buffer.cc
  shape.cc
  storage.cc
  tensor.cc
  tensor_view.cc
)
add_library(inferx::core ALIAS inferx_core)

target_include_directories(inferx_core PUBLIC
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>)

target_link_libraries(inferx_core
  PUBLIC
    inferx::flags
    absl::status
    absl::statusor
    absl::flat_hash_map
    absl::inlined_vector
    absl::span
    absl::strings
  PRIVATE
    mimalloc-static
)
