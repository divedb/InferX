# inferx_server -- the HTTP composition layer around inferx::engine.

add_library(inferx_server
  http_server.cc
)
add_library(inferx::server ALIAS inferx_server)

find_package(OpenSSL REQUIRED)

target_link_libraries(inferx_server
  PUBLIC
    inferx::core
    inferx::support
    inferx::api
    inferx::engine
    inferx::engine_adapters
    inferx::admission
    inferx::auth
    inferx::http_transport
    inferx::managed_request_service
    inferx::model_registry
    inferx::generation
    inferx::request_handlers
    inferx::server_middleware
    inferx::tokenization_service
    inferx::scheduler_client
    inferx::server_observability
    inferx::flags
  PRIVATE
    inferx::beast
    OpenSSL::Crypto
)

if(TARGET inferx_grpc_scheduler_transport)
  target_link_libraries(inferx_server PRIVATE
    inferx::grpc_scheduler_transport)
endif()

# Threads for the bounded HTTP executors.
find_package(Threads REQUIRED)
target_link_libraries(inferx_server PUBLIC Threads::Threads)
