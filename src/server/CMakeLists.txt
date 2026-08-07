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

target_link_libraries(inferx_server
  PUBLIC
    inferx::core
    inferx::support
    inferx::api
    inferx::model
    inferx::observe
    inferx::scheduler
    inferx::tokenizer
    inferx::flags
  PRIVATE
    inferx::beast
)

# Threads, for the engine loop and the bounded HTTP executors.
find_package(Threads REQUIRED)
target_link_libraries(inferx_server PUBLIC Threads::Threads)

add_executable(inferx-serve main.cc)
target_link_libraries(inferx-serve PRIVATE inferx::server)
