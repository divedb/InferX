# Configure-time hardware backend selection.

include_guard(GLOBAL)

set(INFERX_BACKEND "AUTO" CACHE STRING
    "Hardware backend: AUTO, CPU, CUDA, ROCM, or ASCEND")
set_property(CACHE INFERX_BACKEND PROPERTY STRINGS AUTO CPU CUDA ROCM ASCEND)

string(TOUPPER "${INFERX_BACKEND}" _inferx_backend)
set(_inferx_supported_backends AUTO CPU CUDA ROCM ASCEND)
list(FIND _inferx_supported_backends "${_inferx_backend}"
     _inferx_backend_index)
if(_inferx_backend_index EQUAL -1)
  message(FATAL_ERROR
    "Unknown INFERX_BACKEND='${INFERX_BACKEND}'. "
    "Expected one of: AUTO, CPU, CUDA, ROCM, ASCEND")
endif()

# AUTO preserves the legacy INFERX_ENABLE_CUDA switch. Explicit backend
# selection is authoritative so two cache variables cannot disagree.
if(_inferx_backend STREQUAL "AUTO")
  if(INFERX_ENABLE_CUDA)
    set(INFERX_SELECTED_BACKEND "CUDA")
  else()
    set(INFERX_SELECTED_BACKEND "CPU")
  endif()
elseif(_inferx_backend STREQUAL "CPU")
  set(INFERX_SELECTED_BACKEND "CPU")
  set(INFERX_ENABLE_CUDA OFF CACHE BOOL "Build the CUDA device layer" FORCE)
elseif(_inferx_backend STREQUAL "CUDA")
  set(INFERX_SELECTED_BACKEND "CUDA")
  set(INFERX_ENABLE_CUDA ON CACHE BOOL "Build the CUDA device layer" FORCE)
else()
  message(FATAL_ERROR
    "INFERX_BACKEND=${_inferx_backend} is recognized but not implemented yet")
endif()

set(INFERX_SELECTED_BACKEND "${INFERX_SELECTED_BACKEND}" CACHE INTERNAL
    "Resolved InferX hardware backend" FORCE)
