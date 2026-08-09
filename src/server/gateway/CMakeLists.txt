add_library(inferx_gateway_server gateway_server.cc)
add_library(inferx::gateway_server ALIAS inferx_gateway_server)
target_include_directories(inferx_gateway_server PUBLIC
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>)
target_link_libraries(inferx_gateway_server PUBLIC
  inferx::core
  inferx::admission
  inferx::auth
  inferx::grpc_scheduler_transport
  inferx::http_transport
  inferx::managed_request_service
  inferx::model_registry
  inferx::request_handlers
  inferx::server_middleware
  inferx::tokenization_service
  inferx::tokenizer
  inferx::folly_coro)

add_executable(inferx-gateway main.cc)
target_link_libraries(inferx-gateway PRIVATE inferx::gateway_server)
target_link_libraries(inferx-gateway PRIVATE inferx::support)
