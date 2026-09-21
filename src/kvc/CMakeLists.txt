include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
find_package(Threads REQUIRED)

add_library(kvc_abi INTERFACE)
add_library(kvc::abi ALIAS kvc_abi)
set_target_properties(kvc_abi PROPERTIES EXPORT_NAME abi)
target_include_directories(kvc_abi INTERFACE
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

add_library(kvc_runtime STATIC runtime.cc)
add_library(kvc::runtime ALIAS kvc_runtime)
set_target_properties(kvc_runtime PROPERTIES EXPORT_NAME runtime POSITION_INDEPENDENT_CODE ON)
target_compile_features(kvc_runtime PUBLIC cxx_std_23)
target_link_libraries(kvc_runtime PUBLIC kvc::abi PRIVATE ${CMAKE_DL_LIBS} Threads::Threads)
if(NOT MSVC)
  target_compile_options(kvc_runtime PRIVATE -Wall -Wextra -Wpedantic)
endif()

install(TARGETS kvc_abi kvc_runtime EXPORT KvCacheTargets
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(DIRECTORY ${PROJECT_SOURCE_DIR}/include/kvc DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(EXPORT KvCacheTargets NAMESPACE kvc:: DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/KvCache)
configure_package_config_file(${CMAKE_CURRENT_SOURCE_DIR}/KvCacheConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/KvCacheConfig.cmake
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/KvCache)
write_basic_package_version_file(${CMAKE_CURRENT_BINARY_DIR}/KvCacheConfigVersion.cmake
  VERSION 0.1.0 COMPATIBILITY SameMinorVersion)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/KvCacheConfig.cmake
              ${CMAKE_CURRENT_BINARY_DIR}/KvCacheConfigVersion.cmake
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/KvCache)
