cmake_minimum_required(VERSION 3.24)
project(InferX VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(INFERX_BUILD_TESTS "Build the test suite" ON)
option(INFERX_ENABLE_TOKENIZERS "Build tokenizer support" ON)
option(INFERX_ENABLE_CUDA "Build the CUDA backend" ON)
option(INFERX_BUILD_ENGINE "Build the inference engine and its dependencies" ON)
option(INFERX_BUILD_KVCACHE_PLUGINS "Build bundled KV-cache providers" ON)

if(INFERX_BUILD_TESTS)
  enable_testing()
endif()
find_package(Threads REQUIRED)
add_subdirectory(src/kvc)
if(INFERX_BUILD_KVCACHE_PLUGINS)
  add_subdirectory(plugins/kvcache/host)
  if(INFERX_ENABLE_CUDA)
    add_subdirectory(plugins/kvcache/cuda)
  endif()
endif()
if(INFERX_BUILD_TESTS)
  add_subdirectory(tests/kvc)
endif()
# The public provider SDK can be built without any engine dependencies.
if(NOT INFERX_BUILD_ENGINE)
  return()
endif()

include(FetchContent)
set(FETCHCONTENT_QUIET OFF)

# CUDA toolkit: prefer the system install; disable the backend if absent.
if(INFERX_ENABLE_CUDA)
  find_program(INFERX_NVCC nvcc HINTS /usr/local/cuda/bin)
  if(INFERX_NVCC)
    set(CMAKE_CUDA_COMPILER "${INFERX_NVCC}" CACHE FILEPATH "" FORCE)
    set(CMAKE_CUDA_STANDARD 20)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_ARCHITECTURES 89 CACHE STRING "CUDA architectures" FORCE)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
  else()
    message(STATUS "CUDA requested but nvcc was not found; disabling CUDA")
    set(INFERX_ENABLE_CUDA OFF)
  endif()
endif()

# ---------------------------------------------------------------------------
# Well-established third-party dependencies.
# Vendored submodules are used where present; header-only libraries are
# fetched at configure time. All versions are pinned.
# ---------------------------------------------------------------------------

# Abseil: status, statusor, logging, flat_hash_map, flags, spans.
set(ABSL_PROPAGATE_CXX_STD ON CACHE BOOL "" FORCE)
set(ABSL_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
set(ABSL_BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(ABSL_USE_EXTERNAL_GOOGLETEST OFF CACHE BOOL "" FORCE)
set(ABSL_BUILD_TEST_HELPERS OFF CACHE BOOL "" FORCE)
add_subdirectory(third_party/abseil-cpp EXCLUDE_FROM_ALL)

# Host allocator used by the core tensor/storage layer.
set(MI_BUILD_SHARED OFF CACHE BOOL "" FORCE)
set(MI_BUILD_OBJECT OFF CACHE BOOL "" FORCE)
set(MI_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(MI_OVERRIDE OFF CACHE BOOL "" FORCE)
FetchContent_Declare(
  mimalloc
  URL https://github.com/microsoft/mimalloc/archive/refs/tags/v2.1.7.tar.gz
  URL_HASH SHA256=0eed39319f139afde8515010ff59baf24de9e47ea316a315398e8027d198202d
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
FetchContent_MakeAvailable(mimalloc)

# nlohmann/json: configuration and safetensors header parsing.
FetchContent_Declare(
  nlohmann_json
  URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
FetchContent_MakeAvailable(nlohmann_json)

# Google Highway: portable SIMD for CPU elementwise/normalization kernels.
set(HWY_ENABLE_TESTS OFF CACHE BOOL "" FORCE)
set(HWY_ENABLE_EXAMPLES OFF CACHE BOOL "" FORCE)
set(HWY_ENABLE_CONTRIB OFF CACHE BOOL "" FORCE)
set(HWY_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
set(HWY_DISABLED_TARGETS "" CACHE STRING "" FORCE)
add_subdirectory(third_party/highway EXCLUDE_FROM_ALL)

# oneDNN: optimized GEMM for the CPU backend's Linear op.
set(DNNL_LIBRARY_TYPE STATIC CACHE STRING "" FORCE)
set(DNNL_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(DNNL_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(DNNL_BUILD_FOR_CI OFF CACHE BOOL "" FORCE)
set(DNNL_CPU_RUNTIME SEQ CACHE STRING "" FORCE)
set(DNNL_GPU_RUNTIME NONE CACHE STRING "" FORCE)
set(DNNL_ENABLE_JIT_PROFILING OFF CACHE BOOL "" FORCE)
set(DNNL_ENABLE_ITT_TASKS OFF CACHE BOOL "" FORCE)
add_subdirectory(third_party/onednn EXCLUDE_FROM_ALL)

# tokenizers-cpp: HuggingFace JSON tokenizer bindings.
#
# tokenizers-cpp bundles sentencepiece, which in turn bundles its own copy of
# Abseil and conflicts with the top-level Abseil targets. InferX only needs the
# HuggingFace JSON tokenizer path, so it builds the Rust `tokenizers_c` core
# and compiles the single self-contained C++ binding directly, avoiding the
# sentencepiece/Abseil collision entirely. SentencePiece support can be added
# later as an isolated subproject if required.
if(INFERX_ENABLE_TOKENIZERS)
  find_program(CARGO_EXECUTABLE cargo REQUIRED)
  set(TK_SRC ${CMAKE_SOURCE_DIR}/third_party/tokenizers-cpp)
  set(TK_CARGO_TARGET_DIR ${CMAKE_BINARY_DIR}/tokenizers-rust)
  set(TK_RUST_LIB ${CMAKE_BINARY_DIR}/libtokenizers_c.a)
  file(GLOB TK_RUST_SOURCES ${TK_SRC}/rust/src/*.rs ${TK_SRC}/rust/Cargo.toml)

  add_custom_command(
    OUTPUT ${TK_RUST_LIB}
    COMMAND
      ${CMAKE_COMMAND} -E env CARGO_TARGET_DIR=${TK_CARGO_TARGET_DIR}
      ${CARGO_EXECUTABLE} build --release
      --manifest-path ${TK_SRC}/rust/Cargo.toml
    COMMAND
      ${CMAKE_COMMAND} -E copy_if_different
      ${TK_CARGO_TARGET_DIR}/release/libtokenizers_c.a ${TK_RUST_LIB}
    DEPENDS ${TK_RUST_SOURCES}
    COMMENT "Building tokenizers-cpp Rust core"
    VERBATIM
  )
  add_custom_target(tokenizers_rust DEPENDS ${TK_RUST_LIB})

  add_library(inferx_tokenizers STATIC
    ${TK_SRC}/src/huggingface_tokenizer.cc
  )
  target_include_directories(inferx_tokenizers PUBLIC ${TK_SRC}/include)
  add_dependencies(inferx_tokenizers tokenizers_rust)
  target_link_libraries(inferx_tokenizers PUBLIC ${TK_RUST_LIB} ${CMAKE_DL_LIBS})
  target_compile_definitions(inferx_tokenizers PUBLIC INFERX_HAVE_TOKENIZERS)
  target_compile_options(inferx_tokenizers PRIVATE -Wall -Wextra)
endif()

# CLI11: command-line parsing for the `inferx` app (header-only).
set(CLI11_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(CLI11_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(CLI11_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(CLI11_SINGLE_FILE_TESTS OFF CACHE BOOL "" FORCE)
add_subdirectory(third_party/cli11 EXCLUDE_FROM_ALL)

# The engine executes models through FlashInfer's CUDA kernels, so it has no
# CPU-only build; without a CUDA toolchain only the kvc ABI and plugins build.
if(NOT INFERX_ENABLE_CUDA)
  message(STATUS "CUDA unavailable: skipping the inference engine")
  set(INFERX_BUILD_ENGINE OFF)
endif()
if(INFERX_BUILD_ENGINE)
  add_subdirectory(src)
  add_subdirectory(adapters/inferx)

  if(INFERX_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
  endif()
endif()
