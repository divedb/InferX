# inferx_core -- element types, tensors, status, logging, and the memory layer.
#
# Everything here builds with the host compiler. Files that use the CUDA runtime
# API include <cuda_runtime_api.h>, which is plain C; only files containing
# __global__ kernels need nvcc, and there are none at M0.

inferx_add_library(
  NAME core
  SRCS
    cpu/device_runtime.cc
    allocator.cc
    arena.cc
    caching_allocator.cc
    device_buffer.cc
    device_runtime.cc
    logging.cc
    shape.cc
    storage.cc
    tensor.cc
  INCLUDE_DIRS ${PROJECT_SOURCE_DIR}/include
  DEPS
    absl::log
    absl::status
    absl::statusor
    absl::flat_hash_map
    absl::inlined_vector
    absl::span
    absl::strings
  PRIVATE_DEPS absl::log_initialize mimalloc-static
)

if(INFERX_ENABLE_CUDA)
  target_sources(inferx_core PRIVATE cuda/device_runtime.cc)
  # PUBLIC: core/device.h derives kCudaBuilt from it, so every consumer --
  # CLI option validation, config checks -- sees the compile-time capability.
  target_compile_definitions(inferx_core PUBLIC INFERX_WITH_CUDA=1)
  target_link_libraries(inferx_core PRIVATE CUDA::cudart)
else()
  target_sources(inferx_core PRIVATE accelerator_runtime_stub.cc)
endif()
