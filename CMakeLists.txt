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

if(PROJECT_SOURCE_DIR STREQUAL PROJECT_BINARY_DIR)
  message(FATAL_ERROR
    "In-source builds are not allowed. Use: cmake -S . -B build")
endif()

include(InferXOptions)
include(InferXCuda)
include(InferXDependencies)

add_subdirectory(src/core)

if(INFERX_CUDA_MEETS_FLOOR)
  add_subdirectory(src/kernels)
endif()

if(INFERX_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()

message(STATUS "")
message(STATUS "==== InferX ${PROJECT_VERSION} ====")
message(STATUS "  Build type      : ${CMAKE_BUILD_TYPE}")
message(STATUS "  C++ standard    : ${CMAKE_CXX_STANDARD}")
message(STATUS "  Host compiler   : ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
if(INFERX_ENABLE_CUDA)
  message(STATUS "  CUDA toolkit    : ${CUDAToolkit_VERSION}")
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
message(STATUS "  Tests           : ${INFERX_BUILD_TESTS}")
message(STATUS "")
