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
# Asynchronous HTTP foundation -- Boost.Beast 1.91.0 + Folly 2026.08.03.00.
#
# These targets establish the dependency boundary for the active Beast server
# and its coroutine-facing transport contracts. The gitlinks pin exact
# revisions; configuration performs no fetch.
#
# Initialize once before the default configuration:
#   git submodule update --init --checkout third_party/boost third_party/folly
#   git -C third_party/boost submodule update --init --depth 1
#   cmake -S . -B build
# ---------------------------------------------------------------------------
add_library(inferx_boost_beast INTERFACE)
add_library(inferx::beast ALIAS inferx_boost_beast)
add_library(inferx_folly_coro INTERFACE)
add_library(inferx::folly_coro ALIAS inferx_folly_coro)

set(INFERX_BOOST_PIN "1.91.0")
set(INFERX_FOLLY_PIN "2026.08.03.00")
set(INFERX_FAST_FLOAT_PIN "8.0.0")
  _inferx_require_submodule(boost)
  _inferx_require_submodule(folly)
  _inferx_require_submodule(fast_float)

  if(NOT EXISTS "${INFERX_THIRD_PARTY}/boost/libs/beast/include/boost/beast.hpp")
    message(FATAL_ERROR
      "Boost ${INFERX_BOOST_PIN} library submodules are not initialized.\n"
      "Run: git -C third_party/boost submodule update --init --depth 1")
  endif()
  if(NOT EXISTS
     "${INFERX_THIRD_PARTY}/boost/tools/cmake/include/BoostRoot.cmake")
    message(FATAL_ERROR
      "Boost ${INFERX_BOOST_PIN} CMake support is not initialized.\n"
      "Run: git -C third_party/boost submodule update --init --depth 1")
  endif()

  find_package(fmt CONFIG REQUIRED)
  set(FASTFLOAT_INCLUDE_DIR
      "${INFERX_THIRD_PARTY}/fast_float/include"
      CACHE PATH "Pinned fast_float ${INFERX_FAST_FLOAT_PIN} headers" FORCE)

  # Restrict the Boost superproject to the HTTP stack and its discovered
  # dependencies instead of defining targets for every Boost library.
  # Folly unconditionally declares install exports. Those exports cannot
  # include Boost targets embedded from another source tree, and InferX does
  # not currently define an install target of its own. Suppress third-party
  # install rules for this source-build configuration.
  set(CMAKE_SKIP_INSTALL_RULES ON CACHE BOOL
      "Do not export embedded third-party dependency targets" FORCE)
  set(BOOST_INCLUDE_LIBRARIES
      beast asio system context filesystem program_options regex
      CACHE STRING "" FORCE)
  set(BOOST_ENABLE_CMAKE ON CACHE BOOL "" FORCE)
  add_subdirectory("${INFERX_THIRD_PARTY}/boost"
                   "${CMAKE_BINARY_DIR}/third_party/boost"
                   EXCLUDE_FROM_ALL SYSTEM)

  set(_inferx_saved_build_shared_libs ${BUILD_SHARED_LIBS})
  set(_inferx_saved_build_tests ${BUILD_TESTS})
  set(_inferx_saved_build_benchmarks ${BUILD_BENCHMARKS})
  set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
  set(BUILD_TESTS OFF CACHE BOOL "" FORCE)
  set(BUILD_BENCHMARKS OFF CACHE BOOL "" FORCE)
  set(PYTHON_EXTENSIONS OFF CACHE BOOL "" FORCE)
  add_subdirectory("${INFERX_THIRD_PARTY}/folly"
                   "${CMAKE_BINARY_DIR}/third_party/folly"
                   EXCLUDE_FROM_ALL SYSTEM)
  set(BUILD_SHARED_LIBS ${_inferx_saved_build_shared_libs} CACHE BOOL "" FORCE)
  set(BUILD_TESTS ${_inferx_saved_build_tests} CACHE BOOL "" FORCE)
  set(BUILD_BENCHMARKS ${_inferx_saved_build_benchmarks} CACHE BOOL "" FORCE)

  if(NOT TARGET Boost::beast OR NOT TARGET Boost::asio OR
     NOT TARGET Boost::system OR NOT TARGET Boost::context OR
     NOT TARGET Boost::filesystem OR NOT TARGET Boost::program_options OR
     NOT TARGET Boost::regex)
    message(FATAL_ERROR
      "Boost ${INFERX_BOOST_PIN} did not define the targets required by "
      "Beast and Folly")
  endif()
  if(NOT TARGET Folly::folly)
    message(FATAL_ERROR
      "Folly ${INFERX_FOLLY_PIN} did not define Folly::folly")
  endif()

  target_link_libraries(inferx_boost_beast INTERFACE
    Boost::beast
    Boost::asio
    Boost::system)
  target_compile_definitions(inferx_boost_beast INTERFACE
    INFERX_WITH_BOOST_BEAST=1)

  target_link_libraries(inferx_folly_coro INTERFACE Folly::folly)
  target_compile_definitions(inferx_folly_coro INTERFACE
    INFERX_WITH_FOLLY_CORO=1)

message(STATUS
  "HTTP foundation: Boost ${INFERX_BOOST_PIN}, Folly ${INFERX_FOLLY_PIN}")

# ---------------------------------------------------------------------------
# tokenizers-cpp -- universal Hugging Face and SentencePiece backend.
#
# Pinned as a submodule. Its C++ common interface stays behind InferX's own
# tokenizer service so dependency types do not leak into server call sites.
# ---------------------------------------------------------------------------
_inferx_require_submodule(tokenizers-cpp)
set(MLC_ENABLE_SENTENCEPIECE_TOKENIZER ON CACHE BOOL "" FORCE)
set(SPM_ENABLE_SHARED OFF CACHE BOOL "" FORCE)
set(SPM_ENABLE_TCMALLOC OFF CACHE BOOL "" FORCE)
# InferX already builds its pinned Abseil before this dependency. Recent
# SentencePiece otherwise fetches and defines a second Abseil target set.
set(SPM_ABSL_PROVIDER "" CACHE STRING "" FORCE)
set(_inferx_spm_compat "${CMAKE_BINARY_DIR}/sentencepiece-compat")
file(MAKE_DIRECTORY "${_inferx_spm_compat}/third_party")
if(NOT EXISTS "${_inferx_spm_compat}/third_party/absl")
  file(CREATE_LINK "${INFERX_THIRD_PARTY}/abseil-cpp/absl"
                   "${_inferx_spm_compat}/third_party/absl" SYMBOLIC)
endif()
add_subdirectory("${INFERX_THIRD_PARTY}/tokenizers-cpp"
                 EXCLUDE_FROM_ALL SYSTEM)
target_include_directories(sentencepiece-static PRIVATE
  "${_inferx_spm_compat}"
  "${_inferx_spm_compat}/third_party")

# ---------------------------------------------------------------------------
# Planned dependencies, by milestone. Add with:
#   git submodule add --depth 1 <url> third_party/<name>
#
#   M4  simdjson         https://github.com/simdjson/simdjson.git
#                        For the request hot path, if JSON parsing ever shows
#                        up in a profile. It has not yet: a chat request is a
#                        few hundred bytes next to a multi-millisecond step.
#   --  spdlog           https://github.com/gabime/spdlog.git
# ---------------------------------------------------------------------------
