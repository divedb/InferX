# inferx_api -- the OpenAI-compatible wire format, and nothing else.
#
# Host-only, and it must stay that way. This layer is where malformed JSON,
# hostile field types, escaping in generated text and SSE framing are handled,
# which is where an HTTP API's bugs actually are -- and keeping it free of the
# engine is what lets all of that be tested on a machine with no GPU.

add_library(inferx_api
  openai.cc
)
add_library(inferx::api ALIAS inferx_api)

target_link_libraries(inferx_api
  PUBLIC
    inferx::core
    inferx::support
    inferx::tokenizer
    inferx::flags
)
