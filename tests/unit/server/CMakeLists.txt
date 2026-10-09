inferx_add_test(
  NAME server_api_test
  SRCS server_api_test.cc
  DEPS inferx::server nlohmann_json::nlohmann_json
  LABELS unit
)

inferx_add_test(
  NAME event_queue_test
  SRCS event_queue_test.cc
  DEPS inferx::server
  TIMEOUT 30
  LABELS unit
)

inferx_add_test(
  NAME text_delta_test
  SRCS text_delta_test.cc
  DEPS inferx::server
  LABELS unit
)
