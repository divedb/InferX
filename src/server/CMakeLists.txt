# inferx_server -- OpenAI-compatible HTTP ingress/orchestration layer
# (docs/inference_server_design.md). Phase 1: HTTP foundation and lifecycle --
# Asio/Beast HTTP/1.1 server, router, middleware skeleton, config, JSON
# codec, request IDs, health endpoints, /v1/models, keep-alive, limits,
# graceful shutdown. Engine-backed endpoints arrive in later phases.

inferx_add_library(
  NAME server
  HDRS
    ${PROJECT_SOURCE_DIR}/include/inferx/server/api_error_mapper.h
    ${PROJECT_SOURCE_DIR}/include/inferx/server/connection.h
    ${PROJECT_SOURCE_DIR}/include/inferx/server/engine_client.h
    ${PROJECT_SOURCE_DIR}/include/inferx/server/engine_types.h
    ${PROJECT_SOURCE_DIR}/include/inferx/server/http_server.h
    ${PROJECT_SOURCE_DIR}/include/inferx/server/http_types.h
    ${PROJECT_SOURCE_DIR}/include/inferx/server/json_codec.h
    ${PROJECT_SOURCE_DIR}/include/inferx/server/middleware.h
    ${PROJECT_SOURCE_DIR}/include/inferx/server/request_context.h
    ${PROJECT_SOURCE_DIR}/include/inferx/server/routes.h
    ${PROJECT_SOURCE_DIR}/include/inferx/server/router.h
    ${PROJECT_SOURCE_DIR}/include/inferx/server/server_config.h
    ${PROJECT_SOURCE_DIR}/include/inferx/server/services.h
    ${PROJECT_SOURCE_DIR}/include/inferx/server/sse_writer.h
    ${PROJECT_SOURCE_DIR}/include/inferx/server/status.h
  SRCS
    api_error_mapper.cc
    engine_client.cc
    http_server.cc
    http_session.cc
    http_types.cc
    json_codec.cc
    middleware.cc
    request_context.cc
    routes.cc
    router.cc
    server_config.cc
    services.cc
    sse_writer.cc
  INCLUDE_DIRS ${PROJECT_SOURCE_DIR}/include
  DEPS
    inferx::boost
    inferx::beast
    nlohmann_json::nlohmann_json
)
