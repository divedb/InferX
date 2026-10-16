#===============================================================================
# Copyright 2019 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#===============================================================================


####### Expanded from @PACKAGE_INIT@ by configure_package_config_file() #######
####### Any changes to this file will be overwritten by the next CMake run ####
####### The input file was config.cmake.in                            ########

get_filename_component(PACKAGE_PREFIX_DIR "${CMAKE_CURRENT_LIST_DIR}/../../../" ABSOLUTE)

macro(set_and_check _var _file)
  set(${_var} "${_file}")
  if(NOT EXISTS "${_file}")
    message(FATAL_ERROR "File or directory ${_file} referenced by variable ${_var} does not exist !")
  endif()
endmacro()

macro(check_required_components _NAME)
  foreach(comp ${${_NAME}_FIND_COMPONENTS})
    if(NOT ${_NAME}_${comp}_FOUND)
      if(${_NAME}_FIND_REQUIRED_${comp})
        set(${_NAME}_FOUND FALSE)
      endif()
    endif()
  endforeach()
endmacro()

####################################################################################
set(DNNL_CPU_RUNTIME "SEQ")
set(DNNL_CPU_THREADING_RUNTIME "SEQ")
set(DNNL_GPU_RUNTIME "NONE")

set(DNNL_BLAS_VENDOR "NONE")

set(DNNL_GPU_VENDOR "INTEL")

if(DNNL_CPU_THREADING_RUNTIME STREQUAL "TBB")
    find_package(TBB 2017.0 REQUIRED COMPONENTS tbb)
    if(WIN32)
        # On Windows we must link to debug version of TBB library to ensure ABI compatibility
        # with MSVC debug runtime.
        set_property(TARGET TBB::tbb PROPERTY "MAP_IMPORTED_CONFIG_RELWITHMDD" "DEBUG")
    else()
        # On Linux TBB::tbb target may link to libtbb_debug.so which is not compatible with libtbb.so. Linking
        # application to both may result in undefined behavior.
        # See https://uxlfoundation.github.io/oneTBB/main/intro/limitations.html#debug-tbb-in-the-sycl-program
        set_property(TARGET TBB::tbb PROPERTY "MAP_IMPORTED_CONFIG_RELWITHMDD" "RELEASE" "NONE")
        set_property(TARGET TBB::tbb PROPERTY "MAP_IMPORTED_CONFIG_DEBUG" "RELEASE" "NONE")
    endif()
endif()

# Use a custom find module for transitive dependencies
set(DNNL_ORIGINAL_CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH})
list(INSERT CMAKE_MODULE_PATH 0 ${PACKAGE_PREFIX_DIR}/lib/cmake/dnnl)

if(DNNL_BLAS_VENDOR STREQUAL "ACCELERATE")
    set(BLA_VENDOR "Apple")
    find_package(BLAS REQUIRED)
    unset(BLA_VENDOR)
endif()

# Reverting the CMAKE_MODULE_PATH to its original state
set(CMAKE_MODULE_PATH ${DNNL_ORIGINAL_CMAKE_MODULE_PATH})

include("${CMAKE_CURRENT_LIST_DIR}/dnnl-targets.cmake")
check_required_components("dnnl")

if(DNNL_CPU_RUNTIME STREQUAL "SYCL" OR DNNL_CPU_RUNTIME STREQUAL "DPCPP" OR
   DNNL_GPU_RUNTIME STREQUAL "SYCL" OR DNNL_GPU_RUNTIME STREQUAL "DPCPP")
    if(DNNL_GPU_VENDOR STREQUAL "NVIDIA")
        set(DNNL_ORIGINAL_CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH})
        list(INSERT CMAKE_MODULE_PATH 0 ${PACKAGE_PREFIX_DIR}/lib/cmake/dnnl)
        find_package(cuDNN REQUIRED)
        find_package(cuBLAS REQUIRED)
        find_package(cublasLt REQUIRED)
        set(CMAKE_MODULE_PATH ${DNNL_ORIGINAL_CMAKE_MODULE_PATH})
    endif()
    set(DNNL_COMPILE_FLAGS "-fsycl")
    
endif()
