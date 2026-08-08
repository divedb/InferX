add_library(inferx_tokenizer
  tokenizer.cc
  unicode.cc
  chat_template.cc
)
add_library(inferx::tokenizer ALIAS inferx_tokenizer)

target_link_libraries(inferx_tokenizer
  PUBLIC
    inferx::core
    inferx::support
    inferx::flags
  PRIVATE
    tokenizers_cpp
)
