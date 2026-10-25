# Server test suites (docs/server_design.md §13): the mock engine test
# double, its contract conformance tests, config tests, and the boundary /
# dependency-direction guards from §14.1.

inferx_add_library(
  NAME server_test_support
  TEST_ONLY
  SRCS mock_engine.cc
  INCLUDE_DIRS ${CMAKE_CURRENT_LIST_DIR} ${PROJECT_SOURCE_DIR}/include
  DEPS absl::status absl::statusor
  PRIVATE_DEPS Threads::Threads
)

inferx_add_test(
  NAME mock_engine_test
  SRCS mock_engine_test.cc
  DEPS inferx::server_test_support
  LABELS unit
)

inferx_add_test(
  NAME server_config_test
  SRCS config_test.cc
  DEPS inferx::config nlohmann_json::nlohmann_json
  LABELS unit
)

# The engine boundary header must never grow transport or server-side
# includes (§9.1, §14.1 step 5).
add_test(NAME serving_boundary_purity
  COMMAND ${CMAKE_COMMAND}
    -DHEADER=${PROJECT_SOURCE_DIR}/include/inferx/engine/engine_client.h
    -P ${PROJECT_SOURCE_DIR}/cmake/CheckServingBoundary.cmake)
set_tests_properties(serving_boundary_purity PROPERTIES LABELS "unit")

# Dependency direction (§3.5, §14.1 step 3): no engine target may link the
# server library — configuration fails if one ever does.
inferx_assert_not_linked(
  FROM
    inferx_core
    inferx_config
    inferx_cache_manager
    inferx_ops
    inferx_sampling
    inferx_models
    inferx_engine
    inferx_comm
    inferx_dist
    inferx_tokenizer
    inferx_bench
    inferx_diagnostic
  FORBIDDEN inferx_server
)
