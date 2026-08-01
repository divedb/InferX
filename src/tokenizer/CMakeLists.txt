# inferx_tokenizer -- text in, token ids out, and back again.
#
# Host-only, and it stays that way: the tokenizer runs on the request thread,
# never on the device. That is also what lets its conformance suite -- which is
# the only evidence that our token ids match HuggingFace's -- run in CI without
# a GPU.

add_library(inferx_tokenizer
  unicode.cc
  tokenizer.cc
  chat_template.cc
)
add_library(inferx::tokenizer ALIAS inferx_tokenizer)

target_link_libraries(inferx_tokenizer
  PUBLIC
    inferx::core
    inferx::common
    inferx::flags
)
