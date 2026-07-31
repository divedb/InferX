# Third-party dependencies, all vendored as git submodules under third_party/.
#
# Dependencies are added only at the milestone that needs them, so that an early
# checkout stays fast to clone and build. The commented block at the bottom is
# the authoritative record of what is coming and how to add it.

include_guard(GLOBAL)

set(INFERX_THIRD_PARTY "${PROJECT_SOURCE_DIR}/third_party")

function(_inferx_require_submodule name)
  if(NOT EXISTS "${INFERX_THIRD_PARTY}/${name}/CMakeLists.txt")
    message(FATAL_ERROR
      "\n"
      "  Submodule third_party/${name} is missing or empty.\n"
      "  Run: git submodule update --init --recursive\n"
      "  Or:  ./scripts/bootstrap.sh\n")
  endif()
endfunction()

# Third-party code is not held to our warning settings, and must not build its
# own test suites. BUILD_TESTING is restored afterwards for our own targets.
set(_inferx_saved_build_testing ${BUILD_TESTING})
set(BUILD_TESTING OFF CACHE BOOL "" FORCE)

# ---------------------------------------------------------------------------
# Abseil -- Status/StatusOr, flat_hash_map, InlinedVector, synchronization.
# Pinned to LTS 20260526.0.
# ---------------------------------------------------------------------------
_inferx_require_submodule(abseil-cpp)
set(ABSL_PROPAGATE_CXX_STD ON CACHE BOOL "" FORCE)
set(ABSL_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
set(ABSL_BUILD_TESTING OFF CACHE BOOL "" FORCE)
add_subdirectory("${INFERX_THIRD_PARTY}/abseil-cpp" EXCLUDE_FROM_ALL SYSTEM)

# ---------------------------------------------------------------------------
# mimalloc -- backing allocator for HostAllocatorImpl. Pinned to v3.4.3.
#
# Used explicitly through mi_malloc_aligned / mi_free. MI_OVERRIDE is off so it
# leaves the global malloc alone, which keeps it out of the way of ASan/UBSan
# builds (a malloc override would conflict with the sanitizer's own shims).
# ---------------------------------------------------------------------------
_inferx_require_submodule(mimalloc)
set(MI_OVERRIDE     OFF CACHE BOOL "" FORCE)
set(MI_BUILD_SHARED OFF CACHE BOOL "" FORCE)
set(MI_BUILD_OBJECT OFF CACHE BOOL "" FORCE)
set(MI_BUILD_TESTS  OFF CACHE BOOL "" FORCE)
add_subdirectory("${INFERX_THIRD_PARTY}/mimalloc" EXCLUDE_FROM_ALL SYSTEM)

# ---------------------------------------------------------------------------
# GoogleTest -- pinned to v1.17.0.
# ---------------------------------------------------------------------------
if(INFERX_BUILD_TESTS)
  _inferx_require_submodule(googletest)
  set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
  add_subdirectory("${INFERX_THIRD_PARTY}/googletest" EXCLUDE_FROM_ALL SYSTEM)
endif()

set(BUILD_TESTING ${_inferx_saved_build_testing} CACHE BOOL "" FORCE)

# ---------------------------------------------------------------------------
# Planned dependencies, by milestone. Add with:
#   git submodule add --depth 1 <url> third_party/<name>
#
#   M1  cutlass          https://github.com/NVIDIA/cutlass.git
#                        Quantized + grouped GEMM. Requires CUTLASS 4.x for
#                        CUDA 13.x; 3.x is CUDA 12.x only.
#   M3  flashinfer       https://github.com/flashinfer-ai/flashinfer.git
#                        Kernel templates only; we write our own AOT wrappers
#                        against inferx::Tensor. Pinned by commit, not tag.
#   M4  boost (beast)    https://github.com/boostorg/boost.git
#   M4  simdjson         https://github.com/simdjson/simdjson.git
#   M4  tokenizers-cpp   https://github.com/mlc-ai/tokenizers-cpp.git
#   M5  folly            https://github.com/facebook/folly.git
#                        NARROW USE ONLY: MPMCQueue + ProducerConsumerQueue.
#                        See docs/ARCHITECTURE.md section 9.
#   --  spdlog           https://github.com/gabime/spdlog.git
# ---------------------------------------------------------------------------
