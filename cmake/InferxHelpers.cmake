# InferxHelpers.cmake
#
# Target-declaration helpers in the style of Abseil's AbseilHelpers.cmake
# (absl_cc_library), adapted to InferX's conventions:
#   * every library is a target named inferx_<name> with an inferx:: alias,
#   * the C++ standard is fixed project-wide by the top-level CMakeLists
#     (CMAKE_CXX_STANDARD 23, REQUIRED), so no per-target feature is set,
#   * first-party targets compile with -Wall -Wextra -Wpedantic, applied to
#     C++ compilations only so nvcc stays out of the warning set.

include_guard(GLOBAL)

# inferx_add_library()
#
# CMake function imitating Bazel's cc_library rule, following absl_cc_library.
#
# Parameters:
#   NAME:     logical target name (see Note)
#   HDRS:     public header files (also IDE-visible on the target)
#   SRCS:     source files; when no compiled sources are listed, an INTERFACE
#             library is created
#   DEPS:     PUBLIC link dependencies
#   PRIVATE_DEPS:     PRIVATE link dependencies
#   INCLUDE_DIRS:     PUBLIC include directories; relative paths resolve
#                     against the calling directory
#   PRIVATE_INCLUDE_DIRS: PRIVATE include directories (same resolution)
#   DEFINES:          PUBLIC compile definitions
#   PRIVATE_DEFINES:  PRIVATE compile definitions
#   COPTS:            PRIVATE compile options
#   PUBLIC_COPTS:     PUBLIC compile options
#   LINKOPTS:         PRIVATE link options
#   TEST_ONLY:        create the target only when INFERX_BUILD_TESTS is ON
#   NO_WARNINGS:      skip the default -Wall -Wextra -Wpedantic
#
# Note:
#   The real target is always inferx_${NAME} with an alias inferx::${NAME};
#   dependents must use the inferx:: form to keep the namespace clean, and
#   the unprefixed form stays available for same-directory post-processing
#   (conditional target_sources, set_source_files_properties, ...).
#
# Example:
#   inferx_add_library(
#     NAME cache
#     HDRS cache/kv_block_pool.h
#     SRCS cache/kv_block_pool.cc cache/recurrent_state_pool.cc
#     DEPS inferx::core
#   )
function(inferx_add_library)
  cmake_parse_arguments(INFERX_LIB
    "TEST_ONLY;NO_WARNINGS"
    "NAME"
    "HDRS;SRCS;DEPS;PRIVATE_DEPS;INCLUDE_DIRS;PRIVATE_INCLUDE_DIRS;DEFINES;PRIVATE_DEFINES;COPTS;PUBLIC_COPTS;LINKOPTS"
    ${ARGN})

  if(NOT INFERX_LIB_NAME)
    message(FATAL_ERROR "inferx_add_library: NAME is required")
  endif()
  if(INFERX_LIB_UNPARSED_ARGUMENTS)
    message(FATAL_ERROR "inferx_add_library(${INFERX_LIB_NAME}): "
      "unknown arguments: ${INFERX_LIB_UNPARSED_ARGUMENTS}")
  endif()
  if(INFERX_LIB_TEST_ONLY AND NOT INFERX_BUILD_TESTS)
    return()
  endif()

  set(_target inferx_${INFERX_LIB_NAME})

  # Headers listed in SRCS do not turn the library into a compiled one.
  set(_srcs "${INFERX_LIB_SRCS}")
  list(FILTER _srcs EXCLUDE REGEX "\\.(h|hpp|inc)$")

  # Include directories are declared relative to the caller, like sources.
  foreach(_kind INCLUDE_DIRS PRIVATE_INCLUDE_DIRS)
    set(_resolved)
    foreach(_dir IN LISTS INFERX_LIB_${_kind})
      if(NOT IS_ABSOLUTE "${_dir}" AND NOT _dir MATCHES "^\\$<")
        set(_dir "${CMAKE_CURRENT_LIST_DIR}/${_dir}")
      endif()
      list(APPEND _resolved "${_dir}")
    endforeach()
    set(INFERX_LIB_${_kind} "${_resolved}")
  endforeach()

  if(_srcs STREQUAL "")
    # Header-only library: only usage-requirements arguments make sense.
    foreach(_private_arg PRIVATE_DEPS PRIVATE_INCLUDE_DIRS PRIVATE_DEFINES
                             COPTS LINKOPTS)
      if(INFERX_LIB_${_private_arg})
        message(FATAL_ERROR "inferx_add_library(${INFERX_LIB_NAME}): "
          "header-only libraries cannot use ${_private_arg}")
      endif()
    endforeach()
    add_library(${_target} INTERFACE)
    target_include_directories(${_target} INTERFACE ${INFERX_LIB_INCLUDE_DIRS})
    target_link_libraries(${_target} INTERFACE ${INFERX_LIB_DEPS})
    target_compile_definitions(${_target} INTERFACE ${INFERX_LIB_DEFINES})
    target_compile_options(${_target} INTERFACE ${INFERX_LIB_PUBLIC_COPTS})
  else()
    add_library(${_target} "")
    target_sources(${_target} PRIVATE ${INFERX_LIB_SRCS} ${INFERX_LIB_HDRS})
    target_include_directories(${_target} PUBLIC ${INFERX_LIB_INCLUDE_DIRS}
      PRIVATE ${INFERX_LIB_PRIVATE_INCLUDE_DIRS})
    target_link_libraries(${_target}
      PUBLIC ${INFERX_LIB_DEPS}
      PRIVATE ${INFERX_LIB_PRIVATE_DEPS} ${INFERX_LIB_LINKOPTS})
    target_compile_definitions(${_target}
      PUBLIC ${INFERX_LIB_DEFINES}
      PRIVATE ${INFERX_LIB_PRIVATE_DEFINES})
    target_compile_options(${_target}
      PUBLIC ${INFERX_LIB_PUBLIC_COPTS}
      PRIVATE ${INFERX_LIB_COPTS})
    if(NOT INFERX_LIB_NO_WARNINGS)
      target_compile_options(${_target} PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic>)
    endif()
  endif()

  add_library(inferx::${INFERX_LIB_NAME} ALIAS ${_target})
endfunction()
