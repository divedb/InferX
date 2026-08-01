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

# ---------------------------------------------------------------------------
# CUTLASS -- quantized and grouped GEMM. Optional until it is fetched.
#
# Detected rather than required, because it is large and only M1+ needs it: a
# checkout without it builds and tests exactly as before, minus the quantized
# kernels. Fetch it with
#
#   git submodule add --depth 1 https://github.com/NVIDIA/cutlass.git third_party/cutlass
#   git -C third_party/cutlass fetch --depth 1 origin tag v4.6.1
#   git -C third_party/cutlass checkout v4.6.1
#
# (two steps because --depth 1 cannot check out a tag directly, and v4.6.1 is a
# tag rather than a branch.)
#
# Consumed as headers only, never add_subdirectory'd. CUTLASS's own CMake builds
# a profiler and a very large test suite; we want the templates and nothing
# else, which is the same narrowness the Folly note in ARCHITECTURE.md section 9
# argues for. tools/util is included for the reference/host-side helpers the
# kernel conformance tests use.
# ---------------------------------------------------------------------------
add_library(inferx_cutlass INTERFACE)
add_library(inferx::cutlass ALIAS inferx_cutlass)

if(EXISTS "${INFERX_THIRD_PARTY}/cutlass/include/cutlass/cutlass.h")
  target_include_directories(inferx_cutlass SYSTEM INTERFACE
    "${INFERX_THIRD_PARTY}/cutlass/include"
    "${INFERX_THIRD_PARTY}/cutlass/tools/util/include")
  target_compile_definitions(inferx_cutlass INTERFACE INFERX_WITH_CUTLASS=1)
  set(INFERX_HAVE_CUTLASS ON)
  message(STATUS "CUTLASS: found at third_party/cutlass")
else()
  set(INFERX_HAVE_CUTLASS OFF)
  message(STATUS "CUTLASS: not present -- quantized GEMM targets are skipped "
                 "(see cmake/InferXDependencies.cmake to fetch it)")
endif()

# ---------------------------------------------------------------------------
# FlashInfer -- attention kernel templates. Optional, like CUTLASS.
#
# Headers only, and specifically only include/flashinfer/. §9 predicted that its
# wrapper layer would be torch- and JIT-coupled while the template layer is not,
# and that holds: there is no torch include anywhere under include/flashinfer/.
# We compile those templates ahead of time against our own TensorView, which is
# why the pin is by commit rather than by version range -- upgrades are manual
# merges, and that cost is accepted deliberately (§9, R5).
#
# Fetch with:
#   git submodule add --depth 1 https://github.com/flashinfer-ai/flashinfer.git \
#       third_party/flashinfer
#   git -C third_party/flashinfer fetch --depth 1 origin tag v0.6.9
#   git -C third_party/flashinfer checkout v0.6.9
# ---------------------------------------------------------------------------
add_library(inferx_flashinfer INTERFACE)
add_library(inferx::flashinfer ALIAS inferx_flashinfer)

if(EXISTS "${INFERX_THIRD_PARTY}/flashinfer/include/flashinfer/page.cuh")
  target_include_directories(inferx_flashinfer SYSTEM INTERFACE
    "${INFERX_THIRD_PARTY}/flashinfer/include")
  target_compile_definitions(inferx_flashinfer INTERFACE INFERX_WITH_FLASHINFER=1)
  set(INFERX_HAVE_FLASHINFER ON)
  message(STATUS "FlashInfer: found at third_party/flashinfer")
else()
  set(INFERX_HAVE_FLASHINFER OFF)
  message(STATUS "FlashInfer: not present -- the naive paged kernel is used "
                 "(see cmake/InferXDependencies.cmake to fetch it)")
endif()

set(BUILD_TESTING ${_inferx_saved_build_testing} CACHE BOOL "" FORCE)

# ---------------------------------------------------------------------------
# cpp-httplib -- the HTTP/1.1 server behind the OpenAI-compatible API.
#
# §9 nominates Boost.Beast and §5.1 designs the I/O around an Asio io_context
# with C++20 coroutines. This is a deliberate substitution: Beast would have
# meant vendoring the Boost superproject to serve five endpoints, whereas
# cpp-httplib is one header with no build-system footprint at all.
#
# It is checked in as a file rather than wired as a submodule, because it *is*
# a file -- httplib.h and its licence, at v0.18.3. Update by replacing them.
#
# What this defers: thread-per-connection rather than an event loop, so the
# ceiling on concurrent streams is threads rather than descriptors. That
# ceiling is far above what a single 4080 SUPER can serve, since the batch is
# capped by max_running long before the connection count binds.
add_library(inferx_cpp_httplib INTERFACE)
add_library(inferx::cpp_httplib ALIAS inferx_cpp_httplib)

if(EXISTS "${INFERX_THIRD_PARTY}/cpp-httplib/httplib.h")
  target_include_directories(inferx_cpp_httplib SYSTEM INTERFACE
    "${INFERX_THIRD_PARTY}/cpp-httplib")
  set(INFERX_HAVE_HTTPLIB ON)
  message(STATUS "cpp-httplib: found at third_party/cpp-httplib")
else()
  set(INFERX_HAVE_HTTPLIB OFF)
  message(STATUS "cpp-httplib: not present -- inferx-serve will not be built")
endif()

# ---------------------------------------------------------------------------
# Planned dependencies, by milestone. Add with:
#   git submodule add --depth 1 <url> third_party/<name>
#
#   M4  simdjson         https://github.com/simdjson/simdjson.git
#                        For the request hot path, if JSON parsing ever shows
#                        up in a profile. It has not yet: a chat request is a
#                        few hundred bytes next to a multi-millisecond step.
#   M5  folly            https://github.com/facebook/folly.git
#                        NARROW USE ONLY: MPMCQueue + ProducerConsumerQueue.
#                        See docs/ARCHITECTURE.md section 9.
#   --  spdlog           https://github.com/gabime/spdlog.git
# ---------------------------------------------------------------------------
