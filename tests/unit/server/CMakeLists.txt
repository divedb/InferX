# Server unit suites: pure router/config/codec/context tests plus
# loopback-socket suites for the SSE writer and the HTTP server (Phase 1:
# docs/inference_server_design.md §14.2).

inferx_add_test(
  NAME server_router_test
  SRCS router_test.cc
  DEPS inferx::server
  LABELS unit
)

inferx_add_test(
  NAME server_config_test
  SRCS config_test.cc
  DEPS inferx::server nlohmann_json::nlohmann_json
  LABELS unit
)

inferx_add_test(
  NAME server_json_codec_test
  SRCS json_codec_test.cc
  DEPS inferx::server nlohmann_json::nlohmann_json
  LABELS unit
)

inferx_add_test(
  NAME server_request_context_test
  SRCS request_context_test.cc
  DEPS inferx::server
  LABELS unit
)

inferx_add_test(
  NAME server_sse_writer_test
  SRCS sse_writer_test.cc
  DEPS inferx::server
  TIMEOUT 60
  LABELS unit
)

inferx_add_test(
  NAME server_http_test
  SRCS http_server_test.cc
  DEPS inferx::server nlohmann_json::nlohmann_json
  TIMEOUT 120
  LABELS unit
)
