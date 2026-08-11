# Global build options and the shared compile-flags interface target.

include_guard(GLOBAL)

option(INFERX_BUILD_BENCH "Build benchmarks (requires CUDA >= the floor)" ON)
# Compatibility switch used when INFERX_BACKEND=AUTO. Prefer selecting the
# backend explicitly for new build configurations.
option(INFERX_ENABLE_CUDA "Build the CUDA device layer (legacy AUTO selector)" ON)
option(INFERX_ENABLE_NCCL "Build the NCCL communication backend when found" ON)
option(INFERX_ENABLE_GRPC "Build the process-separated scheduler gRPC transport when found" ON)
option(INFERX_WERROR      "Treat warnings as errors" OFF)
option(INFERX_ENABLE_ASAN "Build with AddressSanitizer + UBSan" OFF)

# Escape hatch for the toolchain floor enforced in InferXCuda.cmake. The host
# layer and most of the device layer only use CUDA runtime APIs that are stable
# across 12.x/13.x, so an older toolkit is usable for compile-checking. It is
# not a supported configuration -- see docs/DEVELOPMENT.md.
option(INFERX_ALLOW_UNSUPPORTED_CUDA
       "Permit configuring against a CUDA toolkit older than the supported floor" OFF)

set(INFERX_CUDA_ARCHS "89" CACHE STRING
    "Semicolon-separated CUDA architectures (e.g. \"89\" or \"89;90\")")

# ---------------------------------------------------------------------------
# Language standard. Must be set before third-party subdirectories are added so
# that Abseil and GoogleTest are compiled with a matching ABI.
# ---------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Keep symbols hidden by default; the engine is a single binary, not a library
# with a public ABI, and this measurably shrinks link time.
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN ON)

# ---------------------------------------------------------------------------
# inferx_flags: compiler policy carried by every first-party target, never by
# third_party. Backend libraries must be linked by their concrete owners.
# ---------------------------------------------------------------------------
add_library(inferx_flags INTERFACE)
add_library(inferx::flags ALIAS inferx_flags)

target_compile_options(inferx_flags INTERFACE
  $<$<COMPILE_LANGUAGE:CXX>:
    -fno-omit-frame-pointer
    -Wall
    -Wextra
    -Wpedantic
    -Wshadow
    -Wnon-virtual-dtor
    -Wcast-align
    -Woverloaded-virtual
    -Wno-missing-field-initializers
  >
  $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<BOOL:${INFERX_WERROR}>>:-Werror>
)

if(INFERX_ENABLE_ASAN)
  # Sanitizer flags are global, NOT attached to inferx_flags.
  #
  # Everything else we set deliberately excludes third_party, but sanitizers
  # cannot work that way: Abseil's swisstable enables extra generation-tracking
  # state when it detects a sanitizer build, so compiling our translation units
  # with the flags while Abseil's own .cc files are compiled without them is an
  # ODR violation. It surfaces as a UBSan "load of null pointer" inside
  # raw_hash_set.h -- an abseil-internal report with no bug in our code.
  set(_inferx_san_flags
      "-fsanitize=address,undefined -fno-omit-frame-pointer -fno-sanitize-recover=all")
  string(APPEND CMAKE_C_FLAGS " ${_inferx_san_flags}")
  string(APPEND CMAKE_CXX_FLAGS " ${_inferx_san_flags}")
  string(APPEND CMAKE_EXE_LINKER_FLAGS " -fsanitize=address,undefined")
  string(APPEND CMAKE_SHARED_LINKER_FLAGS " -fsanitize=address,undefined")
endif()
