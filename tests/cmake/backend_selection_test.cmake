cmake_minimum_required(VERSION 3.28)

if(NOT DEFINED INFERX_EXPECTED_BACKEND)
  message(FATAL_ERROR "INFERX_EXPECTED_BACKEND is required")
endif()

include("${CMAKE_CURRENT_LIST_DIR}/../../cmake/InferXBackend.cmake")

if(NOT INFERX_SELECTED_BACKEND STREQUAL INFERX_EXPECTED_BACKEND)
  message(FATAL_ERROR
    "Expected backend ${INFERX_EXPECTED_BACKEND}, got "
    "${INFERX_SELECTED_BACKEND}")
endif()
