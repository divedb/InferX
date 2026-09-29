# inferx_server -- OpenAI-compatible HTTP server (`inferx serve`). The
# gateway owns the engine thread; the coroutine front end runs on
# Boost.Beast/Asio (interface targets from InferXDependencies.cmake); the
# TokenizerPool runs the prompt-preparation worker threads.
inferx_add_library(
  NAME server
  SRCS
    serve.cc
    api.cc
    engine_gateway.cc
    http_server.cc
    tokenizer_pool.cc
  DEPS inferx::engine inferx::boost
  PRIVATE_DEPS
    inferx::beast
    nlohmann_json::nlohmann_json
    absl::flat_hash_map
    absl::status
    absl::statusor
)
