add_executable(inferx-serve main.cc)
target_link_libraries(inferx-serve PRIVATE inferx::server
  inferx::serve_options)
