# The providers use the toolkit from ordinary C++; only the engine needs CUDA
# language support. Keep detected capabilities separate from the user's options.
set(INFERX_HAVE_CUDA_COMPILER OFF)
set(INFERX_HAVE_CUDA_TOOLKIT OFF)
if(NOT INFERX_ENABLE_CUDA)
  return()
endif()
if(NOT INFERX_BUILD_ENGINE AND NOT INFERX_BUILD_KVCACHE_PLUGINS)
  return()
endif()

if(INFERX_BUILD_ENGINE)
  # Preserve the system-install preference of the original build. CMake's PATH
  # search can otherwise select an older distro nvcc alongside /usr/local/cuda.
  # Explicit compiler, environment, and toolkit selections take precedence.
  if(NOT DEFINED CMAKE_CUDA_COMPILER AND "$ENV{CUDACXX}" STREQUAL ""
     AND NOT DEFINED CUDAToolkit_ROOT AND "$ENV{CUDAToolkit_ROOT}" STREQUAL ""
     AND "$ENV{CUDA_PATH}" STREQUAL "")
    find_program(_inferx_system_nvcc nvcc
      PATHS /usr/local/cuda/bin NO_DEFAULT_PATH NO_CACHE)
    if(_inferx_system_nvcc)
      set(CMAKE_CUDA_COMPILER "${_inferx_system_nvcc}" CACHE FILEPATH "CUDA compiler")
    endif()
    unset(_inferx_system_nvcc)
  endif()

  include(CheckLanguage)
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    # Preserve the existing GPU default, but allow presets/toolchains/-D to choose.
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES AND "$ENV{CUDAARCHS}" STREQUAL "")
      set(CMAKE_CUDA_ARCHITECTURES 89 CACHE STRING "CUDA architectures")
    endif()
    set(CMAKE_CUDA_STANDARD 20)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    enable_language(CUDA)

    # Abseil exposes std::source_location in headers included by CUDA sources.
    # Some nvcc/host-library combinations accept C++20 but cannot compile it.
    include(CheckSourceCompiles)
    check_source_compiles(CUDA [[
      #include <source_location>
      void probe(std::source_location loc = std::source_location::current()) {}
      int main() { probe(); }
    ]] INFERX_CUDA_HAS_SOURCE_LOCATION)
    if(NOT INFERX_CUDA_HAS_SOURCE_LOCATION)
      message(FATAL_ERROR
        "The selected CUDA compiler (${CMAKE_CUDA_COMPILER}) cannot compile "
        "C++20 std::source_location, which InferX's Abseil dependency requires. "
        "Configure a fresh build directory with a compatible toolchain using "
        "-DCMAKE_CUDA_COMPILER=/path/to/nvcc (for example, /usr/local/cuda/bin/nvcc).")
    endif()
    set(INFERX_HAVE_CUDA_COMPILER ON)
  endif()
endif()

find_package(CUDAToolkit QUIET)
if(CUDAToolkit_FOUND)
  set(INFERX_HAVE_CUDA_TOOLKIT ON)
else()
  message(STATUS "CUDA toolkit not found: skipping CUDA components")
endif()
