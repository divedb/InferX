# inferx_diagnostic -- numerical debugging tools (`inferx diagnostic *`).
inferx_add_library(
  NAME diagnostic
  SRCS replay_logits.cc
  DEPS inferx::models
  PRIVATE_DEPS nlohmann_json::nlohmann_json
)
