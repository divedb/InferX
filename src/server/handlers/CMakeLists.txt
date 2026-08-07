add_library(inferx_request_handlers
  api_routes.cc
  completion_handler.cc
  embeddings_handler.cc
  health_handler.cc
  models_handler.cc
  tokenize_handler.cc)
add_library(inferx::request_handlers ALIAS inferx_request_handlers)
target_include_directories(inferx_request_handlers PUBLIC
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>)
target_link_libraries(inferx_request_handlers PUBLIC
  inferx::http_transport
  inferx::model_registry
  inferx::scheduler_client
  inferx::server_protocol
  inferx::support
  inferx::tokenization_service
  inferx::flags)
