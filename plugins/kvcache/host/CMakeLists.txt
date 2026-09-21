include(GNUInstallDirs)
# Host reference provider: CPU storage, synchronous copies, terminal-on-
# return transfers. Depends only on the provider SDK headers.
add_library(kvc_host MODULE provider.cc transfer.cc tables.cc)
target_link_libraries(kvc_host PRIVATE kvc::abi)
set_target_properties(kvc_host PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
if(NOT MSVC)
  target_compile_options(kvc_host PRIVATE -Wall -Wextra -Wpedantic -fvisibility=hidden)
else()
  target_compile_options(kvc_host PRIVATE /w)
endif()
install(TARGETS kvc_host LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}/kvcache/providers)
