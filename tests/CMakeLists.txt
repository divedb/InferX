include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/v1.15.2.tar.gz
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# Suites register themselves next to their sources; each call goes through
# the inferx_add_test helper (cmake/InferxHelpers.cmake). Server tests
# (mock engine, boundary checks) live in tests/server per §2.1.
add_subdirectory(unit)
add_subdirectory(integration)
add_subdirectory(regression)
add_subdirectory(server)
