# Command-line parsing lives with the reusable support sources, but remains an
# executable-owned translation unit because its schema uses server types.
add_executable(inferx-serve
  main.cc
  ${PROJECT_SOURCE_DIR}/src/support/options.cc)
target_link_libraries(inferx-serve PRIVATE inferx::server CLI11::CLI11)
