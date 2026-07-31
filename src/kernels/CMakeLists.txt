# inferx_kernels -- device code.
#
# This is the first target in the tree compiled by nvcc. It is added only when
# the toolkit meets INFERX_CUDA_MIN_VERSION (see cmake/InferXCuda.cmake): below
# the floor our core headers do not survive nvcc's frontend, so there is nothing
# to be gained by attempting the build.

add_library(inferx_kernels
  gemm_cublaslt.cc
  quantize.cu
  smoke.cu
)
add_library(inferx::kernels ALIAS inferx_kernels)

target_link_libraries(inferx_kernels
  PUBLIC
    inferx::core
    inferx::flags
  PRIVATE
    # cuBLASLt is an implementation detail: gemm.h is pimpl'd precisely so that
    # including it does not drag cublasLt.h into the bench harness or the tests.
    CUDA::cublasLt
)

# Device code is compiled once per architecture in INFERX_CUDA_ARCHS. Set here
# rather than globally so that host targets are not dragged through the CUDA
# language settings.
set_target_properties(inferx_kernels PROPERTIES
  CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
  CUDA_SEPARABLE_COMPILATION OFF
)
