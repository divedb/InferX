add_library(inferx_auth auth.cc)
add_library(inferx::auth ALIAS inferx_auth)
find_package(OpenSSL REQUIRED)
target_include_directories(inferx_auth PUBLIC
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>)
target_link_libraries(inferx_auth PUBLIC inferx::core inferx::flags
  PRIVATE OpenSSL::Crypto)
