add_executable(crash_stack_example
  unit/support/crash_stack_example.cc)
target_link_libraries(crash_stack_example PRIVATE inferx::support)

# Export executable symbols so the in-process symbolizer can name the example's
# stack frames without requiring a separate debugger.
if(UNIX AND NOT APPLE)
  target_link_options(crash_stack_example PRIVATE -rdynamic)
endif()

set_target_properties(crash_stack_example PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY
    "${PROJECT_BINARY_DIR}/examples/unit/support")
