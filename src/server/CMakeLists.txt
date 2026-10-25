# inferx_server -- the serving frontend library (docs/server_design.md):
# the serve entry seam today; the HTTP core (§14.2) grows here. Exceptions
# stay enabled for this target only (A10, R-3).
#
# The engine boundary header lives with the engine's public headers
# (inferx/engine/engine_client.h, namespace inferx): the server consumes it
# header-only and never links the engine library. The reverse direction is
# enforced at configure time by inferx_assert_not_linked
# (tests/server/CMakeLists.txt).
inferx_add_library(
  NAME server
  HDRS
    ${PROJECT_SOURCE_DIR}/include/inferx/server/serve.h
  SRCS
    serve.cc
  INCLUDE_DIRS ${PROJECT_SOURCE_DIR}/include
  DEPS
    inferx::config
)

# inferx_server_main -- the standalone server binary (§3.5). Phase 0 stub;
# Phase 1 replaces main.cc with the real lifecycle entry.
add_executable(inferx_server_main main.cc)
target_link_libraries(inferx_server_main PRIVATE inferx::server)
target_compile_options(inferx_server_main PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic>)
