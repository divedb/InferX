# Engine dependencies are configured only after its CUDA requirements are met.
# These cache entries provide defaults while preserving caller overrides.

# Abseil: status, statusor, logging, flat_hash_map, flags, spans.
set(ABSL_PROPAGATE_CXX_STD ON CACHE BOOL "")
set(ABSL_ENABLE_INSTALL OFF CACHE BOOL "")
set(ABSL_BUILD_TESTING OFF CACHE BOOL "")
set(ABSL_USE_EXTERNAL_GOOGLETEST OFF CACHE BOOL "")
set(ABSL_BUILD_TEST_HELPERS OFF CACHE BOOL "")
add_subdirectory("${PROJECT_SOURCE_DIR}/third_party/abseil-cpp"
                 "${PROJECT_BINARY_DIR}/third_party/abseil-cpp" EXCLUDE_FROM_ALL)

# Host allocator used by the core tensor/storage layer.
set(MI_BUILD_SHARED OFF CACHE BOOL "")
set(MI_BUILD_OBJECT OFF CACHE BOOL "")
set(MI_BUILD_TESTS OFF CACHE BOOL "")
set(MI_OVERRIDE OFF CACHE BOOL "")
add_subdirectory("${PROJECT_SOURCE_DIR}/third_party/mimalloc"
                 "${PROJECT_BINARY_DIR}/third_party/mimalloc" EXCLUDE_FROM_ALL)

# nlohmann/json: configuration and safetensors header parsing.
set(JSON_BuildTests OFF CACHE BOOL "Build nlohmann/json tests")
set(JSON_Install OFF CACHE BOOL "Install nlohmann/json")
add_subdirectory("${PROJECT_SOURCE_DIR}/third_party/nlohmann_json"
                 "${PROJECT_BINARY_DIR}/third_party/nlohmann_json" EXCLUDE_FROM_ALL)

# Google Highway: portable SIMD for CPU elementwise/normalization kernels.
set(HWY_ENABLE_TESTS OFF CACHE BOOL "")
set(HWY_ENABLE_EXAMPLES OFF CACHE BOOL "")
set(HWY_ENABLE_CONTRIB OFF CACHE BOOL "")
set(HWY_ENABLE_INSTALL OFF CACHE BOOL "")
set(HWY_DISABLED_TARGETS "" CACHE STRING "")
add_subdirectory("${PROJECT_SOURCE_DIR}/third_party/highway"
                 "${PROJECT_BINARY_DIR}/third_party/highway" EXCLUDE_FROM_ALL)

# oneDNN: optimized GEMM for the CPU backend's Linear op.
set(DNNL_LIBRARY_TYPE STATIC CACHE STRING "")
set(DNNL_BUILD_TESTS OFF CACHE BOOL "")
set(DNNL_BUILD_EXAMPLES OFF CACHE BOOL "")
set(DNNL_BUILD_FOR_CI OFF CACHE BOOL "")
set(DNNL_CPU_RUNTIME SEQ CACHE STRING "")
set(DNNL_GPU_RUNTIME NONE CACHE STRING "")
set(DNNL_ENABLE_JIT_PROFILING OFF CACHE BOOL "")
set(DNNL_ENABLE_ITT_TASKS OFF CACHE BOOL "")
add_subdirectory("${PROJECT_SOURCE_DIR}/third_party/onednn"
                 "${PROJECT_BINARY_DIR}/third_party/onednn" EXCLUDE_FROM_ALL)

# CLI11: command-line parsing for the `inferx` app (header-only).
set(CLI11_BUILD_TESTS OFF CACHE BOOL "")
set(CLI11_BUILD_EXAMPLES OFF CACHE BOOL "")
set(CLI11_BUILD_DOCS OFF CACHE BOOL "")
set(CLI11_SINGLE_FILE_TESTS OFF CACHE BOOL "")
add_subdirectory("${PROJECT_SOURCE_DIR}/third_party/cli11"
                 "${PROJECT_BINARY_DIR}/third_party/cli11" EXCLUDE_FROM_ALL)

# Boost.Asio subset + Beast: coroutine-based HTTP for the serving layer
# (`inferx serve`). Header-only interface targets; the Boost subset's
# provenance is pinned in third_party/boost/fetch.sh.
find_package(Threads REQUIRED)
add_library(inferx_boost INTERFACE)
target_include_directories(inferx_boost INTERFACE
  "${PROJECT_SOURCE_DIR}/third_party/boost/include")
target_link_libraries(inferx_boost INTERFACE Threads::Threads)
add_library(inferx::boost ALIAS inferx_boost)

add_library(inferx_beast INTERFACE)
target_include_directories(inferx_beast INTERFACE
  "${PROJECT_SOURCE_DIR}/third_party/beast/include")
target_link_libraries(inferx_beast INTERFACE inferx::boost)
add_library(inferx::beast ALIAS inferx_beast)
