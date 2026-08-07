# Compatibility bridge for dependencies that still use CMake's FindBoost
# module after InferX has added the pinned Boost superproject.

if(NOT TARGET Boost::headers)
  include("${CMAKE_ROOT}/Modules/FindBoost.cmake")
  return()
endif()

# FindBoost traditionally returns one merged include tree. A Boost source
# superproject instead gives each modular target its own include directory.
# Folly still includes headers from modules that are not explicit link
# components, so reproduce the merged-tree semantics for this compatibility
# path.
file(GLOB Boost_INCLUDE_DIRS LIST_DIRECTORIES TRUE
     "${Boost_SOURCE_DIR}/libs/*/include"
     "${Boost_SOURCE_DIR}/libs/numeric/*/include")
list(APPEND Boost_INCLUDE_DIRS
     "${Boost_SOURCE_DIR}/libs/headers/include")
list(REMOVE_DUPLICATES Boost_INCLUDE_DIRS)
set(Boost_INCLUDE_DIR "${Boost_INCLUDE_DIRS}")
set(Boost_VERSION_STRING "1.91.0")
set(Boost_VERSION 109100)

foreach(_boost_component IN LISTS Boost_FIND_COMPONENTS)
  if(TARGET "Boost::${_boost_component}")
    set("Boost_${_boost_component}_FOUND" TRUE)
    set("Boost_${_boost_component}_LIBRARY" "Boost::${_boost_component}")
    set("Boost_${_boost_component}_LIBRARIES" "Boost::${_boost_component}")
  else()
    set("Boost_${_boost_component}_FOUND" FALSE)
  endif()
endforeach()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
  Boost
  REQUIRED_VARS Boost_INCLUDE_DIRS
  VERSION_VAR Boost_VERSION_STRING
  HANDLE_COMPONENTS)

set(Boost_LIBRARIES "")
foreach(_boost_component IN LISTS Boost_FIND_COMPONENTS)
  if(Boost_${_boost_component}_FOUND)
    list(APPEND Boost_LIBRARIES "Boost::${_boost_component}")
  endif()
endforeach()
