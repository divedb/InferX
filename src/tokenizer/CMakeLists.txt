# inferx_tokenizer -- the tokenizer subsystem (docs/tokenizer_design.md):
# the HFTokenizer facade over tokenizers-cpp, metadata merging from the
# HuggingFace config side, minja-based chat-template rendering, and
# ModelConfig-driven tokenizer resolution. The tokenizers-cpp backend is
# linked when INFERX_ENABLE_TOKENIZERS is on; without it HFTokenizer
# construction fails with Unimplemented and the target still links.
# minja and nlohmann stay private: they are visible only through the
# subsystem's internal headers, never through include/inferx/tokenizer.
inferx_add_library(
  NAME tokenizer
  SRCS
    chat_template_renderer.cc
    hf_tokenizer.cc
    rust_tokenizer.cc
    tokenizer_metadata.cc
    tokenizer_resolver.cc
  DEPS inferx::core
  PRIVATE_DEPS inferx::minja nlohmann_json::nlohmann_json
)

if(INFERX_ENABLE_TOKENIZERS)
  target_link_libraries(inferx_tokenizer PUBLIC inferx::tokenizers)
endif()
