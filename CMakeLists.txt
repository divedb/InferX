# InferX — high-performance C++20 LLM inference engine.
#
# CMake 3.28 is the floor because that is what Ubuntu 24.04 ships. 3.31+ is
# recommended when building against CUDA 13.x, which knows about newer
# architectures (sm_100/sm_120) that older CMake will refuse to pass through.

cmake_minimum_required(VERSION 3.28)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")

project(inferx
  VERSION 0.1.0
  DESCRIPTION "High-performance C++20 LLM inference engine"
  LANGUAGES CXX)

include(CTest)

if(PROJECT_SOURCE_DIR STREQUAL PROJECT_BINARY_DIR)
  message(FATAL_ERROR
    "In-source builds are not allowed. Use: cmake -S . -B build")
endif()

include(InferXOptions)
include(InferXBackend)
include(InferXCuda)
include(InferXDependencies)

set(INFERX_HAS_DEVICE_OPS OFF)
if(INFERX_SELECTED_BACKEND STREQUAL "CUDA" AND INFERX_CUDA_MEETS_FLOOR)
  set(INFERX_HAS_DEVICE_OPS ON)
endif()

add_subdirectory(src/support)
add_subdirectory(src/observe)
add_subdirectory(src/core)
add_subdirectory(src/comm)
add_subdirectory(src/tokenizer)

if(INFERX_HAS_DEVICE_OPS)
  if(INFERX_SELECTED_BACKEND STREQUAL "CUDA")
    add_subdirectory(src/backends/cuda/ops)
  else()
    message(FATAL_ERROR
      "No ops build dispatch for backend ${INFERX_SELECTED_BACKEND}")
  endif()
endif()

# After the selected ops backend: the model's forward pass links inferx::ops
# when device
# code is available, and is loader-only when it is not.
add_subdirectory(src/model)
add_subdirectory(src/scheduler)
add_subdirectory(src/api)
add_subdirectory(src/server/api)
add_subdirectory(src/server/admission)
add_subdirectory(src/server/auth)
add_subdirectory(src/server/config)
add_subdirectory(src/server/model_registry)
add_subdirectory(src/server/generation)
add_subdirectory(src/server/request)
add_subdirectory(src/server/scheduler_client)
add_subdirectory(src/server/request_service)
add_subdirectory(src/server/streaming)
add_subdirectory(src/server/tokenization)
add_subdirectory(src/server/transport)
add_subdirectory(src/server/middleware)
add_subdirectory(src/server/observability)
add_subdirectory(src/server/handlers)
if(TARGET inferx_grpc_scheduler_transport)
  add_subdirectory(src/server/gateway)
endif()

# The server owns the model, so it needs device code; the wire format it speaks
# does not, which is why src/api is unconditional and this is not.
if(INFERX_HAS_DEVICE_OPS)
  add_subdirectory(src/engine)
  add_subdirectory(src/server)
  add_subdirectory(src/main)
endif()

if(INFERX_BUILD_BENCH AND INFERX_HAS_DEVICE_OPS)
  add_subdirectory(bench)
endif()

# Demos exercise FlashInfer kernel templates directly (no PyTorch/TVM-FFI layer).
# Gated on both the CUDA floor and FlashInfer presence (see demo/CMakeLists.txt).
add_subdirectory(demo)

# Small, host-only examples of reusable InferX components.
add_subdirectory(examples)

if(BUILD_TESTING)
  add_subdirectory(tests)
endif()

message(STATUS "")
message(STATUS "==== InferX ${PROJECT_VERSION} ====")
message(STATUS "  Build type      : ${CMAKE_BUILD_TYPE}")
message(STATUS "  C++ standard    : ${CMAKE_CXX_STANDARD}")
message(STATUS "  Host compiler   : ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "  Backend         : ${INFERX_SELECTED_BACKEND}")
if(INFERX_ENABLE_CUDA)
  message(STATUS "  CUDA toolkit    : ${CUDAToolkit_VERSION} (${CUDAToolkit_BIN_DIR})")
  # Reported separately from the toolkit, and with its path, because these are
  # resolved independently and a build that silently used a different nvcc than
  # the one it validated is exactly the failure this summary should expose.
  message(STATUS "  CUDA compiler   : ${CMAKE_CUDA_COMPILER_VERSION} (${CMAKE_CUDA_COMPILER})")
  message(STATUS "  CUDA archs      : ${CMAKE_CUDA_ARCHITECTURES}")
  if(INFERX_CUDA_MEETS_FLOOR)
    message(STATUS "  Device code     : ENABLED")
  else()
    message(STATUS "  Device code     : SKIPPED "
                   "(toolkit below ${INFERX_CUDA_MIN_VERSION})")
  endif()
else()
  message(STATUS "  CUDA            : DISABLED (host-only build)")
endif()
message(STATUS "  HTTP transport  : Boost.Beast + Boost.Asio")
message(STATUS "")
