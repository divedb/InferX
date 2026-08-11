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

# ---------------------------------------------------------------------------
# Toolkit selection.
#
# Two things resolve independently and are free to disagree: find_package()
# below locates headers and libraries, while enable_language(CUDA) picks the
# compiler off PATH. A box with a distro nvidia-cuda-toolkit installed has
# /usr/bin/nvcc shadowing /usr/local/cuda/bin/nvcc, and the result is a
# configuration that passes a version gate on the toolkit and then compiles
# every .cu with a different, older nvcc -- while the summary reports the
# version that was checked rather than the one that will run.
#
# So point both at the same place before either resolves. Only when the user
# has expressed no preference: an explicit -DCMAKE_CUDA_COMPILER,
# -DCUDAToolkit_ROOT, or the CUDACXX/CUDAToolkit_ROOT environment variables all
# win over this, which is what makes a second toolkit testable.
# ---------------------------------------------------------------------------
set(INFERX_CUDA_ROOT_HINT "/usr/local/cuda" CACHE PATH
    "Toolkit to use when neither CMAKE_CUDA_COMPILER nor CUDAToolkit_ROOT is set")

if(NOT DEFINED CMAKE_CUDA_COMPILER
   AND NOT DEFINED CUDAToolkit_ROOT
   AND NOT DEFINED ENV{CUDAToolkit_ROOT}
   AND NOT DEFINED ENV{CUDACXX}
   AND EXISTS "${INFERX_CUDA_ROOT_HINT}/bin/nvcc")
  set(CUDAToolkit_ROOT "${INFERX_CUDA_ROOT_HINT}" CACHE PATH
      "CUDA toolkit root")
  set(CMAKE_CUDA_COMPILER "${INFERX_CUDA_ROOT_HINT}/bin/nvcc" CACHE FILEPATH
      "CUDA compiler")
  message(STATUS "CUDA toolkit defaulted to ${INFERX_CUDA_ROOT_HINT} "
                 "(override with -DINFERX_CUDA_ROOT_HINT=... or -DCUDAToolkit_ROOT=...)")
endif()

# Locate the toolkit before enable_language() so a bad version produces a clear
# diagnostic rather than an opaque compiler-detection failure. This is the early
# check; the authoritative one runs against the compiler further down.
find_package(CUDAToolkit REQUIRED)

# The guidance every version diagnostic below ends with. Kept in one place so
# the pre-flight and the authoritative check cannot drift apart in what they
# tell the user to do.
set(_inferx_cuda_remedy
    "  Install CUDA 13.x:   ./scripts/install-cuda.sh\n"
    "  Select a toolkit:    cmake -S . -B build -DINFERX_CUDA_ROOT_HINT=/usr/local/cuda-13.0\n"
    "  Build without CUDA:  cmake -S . -B build -DINFERX_ENABLE_CUDA=OFF\n"
    "  Override (unsupported, host layer only):\n"
    "                       cmake -S . -B build -DINFERX_ALLOW_UNSUPPORTED_CUDA=ON\n")

# Pre-flight. Not authoritative -- it reports on the toolkit find_package()
# located, which is not necessarily the compiler that will build anything. It
# exists to fail early and legibly in the common case where both agree.
if(CUDAToolkit_VERSION VERSION_LESS INFERX_CUDA_MIN_VERSION
   AND NOT INFERX_ALLOW_UNSUPPORTED_CUDA)
  message(FATAL_ERROR
    "\n"
    "  InferX requires CUDA >= ${INFERX_CUDA_MIN_VERSION}, found ${CUDAToolkit_VERSION}\n"
    "  at ${CUDAToolkit_BIN_DIR}.\n"
    "\n"
    ${_inferx_cuda_remedy})
endif()

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)
set(CMAKE_CUDA_SEPARABLE_COMPILATION OFF)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES "${INFERX_CUDA_ARCHS}")
endif()

enable_language(CUDA)

# ---------------------------------------------------------------------------
# The authoritative checks. Everything above ran against the toolkit that
# find_package() located; CMAKE_CUDA_COMPILER_VERSION is the program that will
# actually compile every .cu, and it is the only version that can be trusted to
# describe the build.
# ---------------------------------------------------------------------------

# Headers and libraries from one toolkit with a compiler from another is not a
# supported configuration even when both clear the floor -- it is how a build
# ends up reporting a version it is not using. Compared on major.minor, since
# that is the granularity at which the two are expected to agree.
string(REGEX MATCH "^[0-9]+\\.[0-9]+" _inferx_tk_mm "${CUDAToolkit_VERSION}")
string(REGEX MATCH "^[0-9]+\\.[0-9]+" _inferx_cc_mm "${CMAKE_CUDA_COMPILER_VERSION}")

if(NOT _inferx_tk_mm VERSION_EQUAL _inferx_cc_mm)
  message(FATAL_ERROR
    "\n"
    "  CUDA toolkit and compiler are from different installations.\n"
    "\n"
    "    toolkit  ${CUDAToolkit_VERSION}  (${CUDAToolkit_BIN_DIR})\n"
    "    compiler ${CMAKE_CUDA_COMPILER_VERSION}  (${CMAKE_CUDA_COMPILER})\n"
    "\n"
    "  This usually means a distro nvidia-cuda-toolkit at /usr/bin/nvcc is\n"
    "  shadowing the real one on PATH. Point both at the same root:\n"
    "\n"
    ${_inferx_cuda_remedy})
endif()

if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS INFERX_CUDA_MIN_VERSION)
  if(INFERX_ALLOW_UNSUPPORTED_CUDA)
    message(WARNING
      "\n"
      "  Compiling with nvcc ${CMAKE_CUDA_COMPILER_VERSION}, which is BELOW the\n"
      "  supported floor of ${INFERX_CUDA_MIN_VERSION}.\n"
      "\n"
      "  This is a compile-check configuration only. It builds the host layer\n"
      "  against the CUDA runtime API; every target containing device code is\n"
      "  SKIPPED, starting with src/backends/cuda/ops.\n"
      "\n"
      "  Install CUDA 13.x: ./scripts/install-cuda.sh\n")
  else()
    message(FATAL_ERROR
      "\n"
      "  InferX requires nvcc >= ${INFERX_CUDA_MIN_VERSION}, found "
      "${CMAKE_CUDA_COMPILER_VERSION}\n"
      "  at ${CMAKE_CUDA_COMPILER}.\n"
      "\n"
      ${_inferx_cuda_remedy})
  endif()
else()
  set(INFERX_CUDA_MEETS_FLOOR ON)
endif()

# nvcc 12.0-12.3 rejects libstdc++ 13 headers. Only reachable under the escape
# hatch now, but say so up front rather than letting it surface as 400 lines of
# template errors from inside libstdc++.
if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS "12.4"
   AND CMAKE_CXX_COMPILER_ID STREQUAL "GNU"
   AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL "13")
  message(WARNING
    "\n"
    "  nvcc ${CMAKE_CUDA_COMPILER_VERSION} does not support GCC "
    "${CMAKE_CXX_COMPILER_VERSION} as a host compiler.\n"
    "  Device compilation will likely fail inside libstdc++ headers.\n"
    "\n"
    "  Workaround: -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-12\n"
    "  Real fix:   install CUDA 13.x\n")
endif()

target_compile_definitions(inferx_flags INTERFACE INFERX_WITH_CUDA=1)

target_compile_options(inferx_flags INTERFACE
  $<$<COMPILE_LANGUAGE:CUDA>:
    --expt-relaxed-constexpr
    --expt-extended-lambda
    -lineinfo
    $<$<BOOL:${INFERX_WERROR}>:-Werror=all-warnings>
  >
)
