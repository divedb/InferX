# inferx_server -- the engine's step loop and the HTTP front end.
#
# Unlike inferx::api, this needs a device: it owns the model. Anything that can
# be tested without one belongs in inferx::api instead, which is why the wire
# format lives there and only the loop and the socket live here.

add_library(inferx_server
  engine.cc
  http_server.cc
  qwen_runner.cc
)
add_library(inferx::server ALIAS inferx_server)

find_package(OpenSSL REQUIRED)

target_link_libraries(inferx_server
  PUBLIC
    inferx::core
    inferx::support
    inferx::api
    inferx::model
    inferx::observe
    inferx::engine_client
    inferx::admission
    inferx::auth
    inferx::http_transport
    inferx::managed_request_service
    inferx::model_registry
    inferx::request_handlers
    inferx::server_middleware
    inferx::tokenization_service
    inferx::scheduler_client
    inferx::server_observability
    inferx::scheduler
    inferx::tokenizer
    inferx::flags
  PRIVATE
    inferx::beast
    OpenSSL::Crypto
)

# Threads, for the engine loop and the bounded HTTP executors.
find_package(Threads REQUIRED)
target_link_libraries(inferx_server PUBLIC Threads::Threads)

add_executable(inferx-serve main.cc)
target_link_libraries(inferx-serve PRIVATE inferx::server)
