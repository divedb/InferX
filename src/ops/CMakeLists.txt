# inferx_ops -- the portable compute-op layer. Validation and dispatch are
# ordinary C++; each backend directory (ops/cpu, ops/cuda) is compiled
# separately and keeps its library-specific details to itself. The CPU
# backend's Highway kernels are compiled once per SIMD target and picked at
# runtime; the CUDA backend alone sees FlashInfer and nvcc.
inferx_add_library(
  NAME ops
  SRCS
    norm.cc
    cpu/norm.cc
    cuda/norm.cu
    gather.cc
    linear.cc
    elementwise.cc
    rotary.cc
    attention.cc
    moe.cc
    mla.cc
    gdn.cc
    flash_attention.cc
    cuda/flash_attention.cu
    cuda/generic_attention.cu
    cuda/moe.cu
    cuda/mla.cu
    cuda/gdn.cu
    cuda/sampling.cu
    cuda/gather.cu
    cuda/linear.cu
    cuda/elementwise.cu
    cuda/rotary.cu
    cuda/attention.cu
    cuda/cublas.cu
  DEPS inferx::core
  PRIVATE_DEPS CUDA::cudart CUDA::cublas hwy
  PRIVATE_INCLUDE_DIRS
    ${PROJECT_SOURCE_DIR}/third_party/flashinfer/include
    ${PROJECT_SOURCE_DIR}/third_party/flashinfer/3rdparty/cccl/libcudacxx/include
    ${PROJECT_SOURCE_DIR}/third_party/flashinfer/3rdparty/cutlass/include
)

set_source_files_properties(cuda/flash_attention.cu PROPERTIES
  COMPILE_OPTIONS "--expt-relaxed-constexpr")
# Highway re-includes norm.cc once per SIMD target through
# HWY_TARGET_INCLUDE ("ops/cpu/norm.cc"), which resolves against the
# include path, so it points at src/, not this directory.
set_source_files_properties(cpu/norm.cc PROPERTIES
  COMPILE_OPTIONS "-I${PROJECT_SOURCE_DIR}/src")

# FlashInfer LayerNorm gates its BF16 type traits on this definition.
set_source_files_properties(cuda/norm.cu PROPERTIES
  COMPILE_DEFINITIONS "ENABLE_BF16=1")

# Unified attention architecture (host side): spec, bridge, capability,
# cache views. See docs/attention.md.
add_subdirectory(attention)
