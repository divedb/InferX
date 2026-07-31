# CUDA toolchain discovery, version enforcement, and language enablement.

include_guard(GLOBAL)

# The supported floor. CUDA 13.x is required for the kernel work that lands in
# M1+: CUTLASS 4.x FP8/grouped-GEMM paths and the FlashInfer MLA wrappers both
# assume it, and 13.x is the first toolkit that officially supports GCC 13/14 as
# a host compiler. See docs/ARCHITECTURE.md risk R1.
set(INFERX_CUDA_MIN_VERSION "13.0")

# Whether device code (nvcc-compiled targets) can be built at all. Distinct from
# INFERX_ENABLE_CUDA, which only says whether the CUDA runtime API is available
# to host translation units. Below the floor the two diverge: the host layer
# still builds and links against cudart, but nothing can go through nvcc,
# because our core headers do not survive its frontend there -- absl's
# status.h uses std::source_location, which nvcc 12.0 cannot evaluate. Targets
# containing .cu files must be gated on this, not on INFERX_ENABLE_CUDA.
set(INFERX_CUDA_MEETS_FLOOR OFF)

if(NOT INFERX_ENABLE_CUDA)
  message(STATUS "CUDA disabled: building host-only core "
                 "(INFERX_WITH_CUDA will not be defined)")
  return()
endif()

# Locate the toolkit before enable_language() so a bad version produces a clear
# diagnostic rather than an opaque compiler-detection failure.
find_package(CUDAToolkit REQUIRED)

if(CUDAToolkit_VERSION VERSION_LESS INFERX_CUDA_MIN_VERSION)
  if(INFERX_ALLOW_UNSUPPORTED_CUDA)
    message(WARNING
      "\n"
      "  Configuring against CUDA ${CUDAToolkit_VERSION}, which is BELOW the\n"
      "  supported floor of ${INFERX_CUDA_MIN_VERSION}.\n"
      "\n"
      "  This is a compile-check configuration only. It builds the host layer\n"
      "  against the CUDA runtime API; every target containing device code is\n"
      "  SKIPPED, starting with src/kernels.\n"
      "\n"
      "  Install CUDA 13.x: ./scripts/install-cuda.sh\n")
  else()
    message(FATAL_ERROR
      "\n"
      "  InferX requires CUDA >= ${INFERX_CUDA_MIN_VERSION}, found ${CUDAToolkit_VERSION}\n"
      "  at ${CUDAToolkit_BIN_DIR}.\n"
      "\n"
      "  Install CUDA 13.x:   ./scripts/install-cuda.sh\n"
      "  Build without CUDA:  cmake -S . -B build -DINFERX_ENABLE_CUDA=OFF\n"
      "  Override (unsupported, compile-check only):\n"
      "                       cmake -S . -B build -DINFERX_ALLOW_UNSUPPORTED_CUDA=ON\n")
  endif()
else()
  set(INFERX_CUDA_MEETS_FLOOR ON)
endif()

# nvcc 12.0-12.3 rejects libstdc++ 13 headers. If the user is on that pairing,
# say so up front rather than letting them read 400 lines of template errors.
if(CUDAToolkit_VERSION VERSION_LESS "12.4"
   AND CMAKE_CXX_COMPILER_ID STREQUAL "GNU"
   AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL "13")
  message(WARNING
    "\n"
    "  CUDA ${CUDAToolkit_VERSION} does not support GCC "
    "${CMAKE_CXX_COMPILER_VERSION} as a host compiler.\n"
    "  Device compilation will likely fail inside libstdc++ headers.\n"
    "\n"
    "  Workaround: -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-12\n"
    "  Real fix:   install CUDA 13.x\n")
endif()

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)
set(CMAKE_CUDA_SEPARABLE_COMPILATION OFF)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES "${INFERX_CUDA_ARCHS}")
endif()

enable_language(CUDA)

target_compile_definitions(inferx_flags INTERFACE INFERX_WITH_CUDA=1)

target_compile_options(inferx_flags INTERFACE
  $<$<COMPILE_LANGUAGE:CUDA>:
    --expt-relaxed-constexpr
    --expt-extended-lambda
    -lineinfo
    $<$<BOOL:${INFERX_WERROR}>:-Werror=all-warnings>
  >
)

target_link_libraries(inferx_flags INTERFACE CUDA::cudart)
