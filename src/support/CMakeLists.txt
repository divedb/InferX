# inferx_support -- small utilities with no device or model dependencies.

add_library(inferx_support
  crash.cc
  file.cc
  json.cc
  log.cc
  shutdown.cc
)
add_library(inferx::support ALIAS inferx_support)

target_include_directories(inferx_support PUBLIC
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>)

target_link_libraries(inferx_support
  PUBLIC
    inferx::flags
    absl::status
    absl::statusor
    absl::strings
    absl::log
    absl::log_initialize
    absl::log_globals
    absl::log_sink_registry
    absl::time
    absl::failure_signal_handler
    absl::symbolize
)

# inferx_vllm_compat -- the vLLM flag-stub registry, shared by both binaries'
# parsers and their tests. Separate from inferx_support so that lib keeps no
# CLI11 or OpenSSL dependency.
find_package(OpenSSL REQUIRED)
add_library(inferx_vllm_compat vllm_compat.cc)
add_library(inferx::vllm_compat ALIAS inferx_vllm_compat)
target_link_libraries(inferx_vllm_compat
  PUBLIC inferx::support CLI11::CLI11
  PRIVATE OpenSSL::Crypto)

# inferx_serve_options -- inferx-serve's CLI schema as a linkable library, so
# the parser is unit-testable. Public deps carry the header surface of
# ServeOptions (engine + http-server config types).
add_library(inferx_serve_options options.cc)
add_library(inferx::serve_options ALIAS inferx_serve_options)
target_link_libraries(inferx_serve_options
  PUBLIC inferx::vllm_compat inferx::core inferx::folly_coro)
