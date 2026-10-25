# Tokenizer subsystem tests: they exercise the real tokenizers-cpp backend
# and the minja renderer, so they are registered only when the backend is
# built. Fixtures are checked out under tests/testdata and referenced
# relative to the project root.
if(INFERX_ENABLE_TOKENIZERS)
  inferx_add_test(
    NAME hf_tokenizer_test
    SRCS hf_tokenizer_test.cc
    DEPS inferx::tokenizer
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
    LABELS unit
  )

  # Golden parity: renders through the fixture's real chat template and
  # compares against transformers-captured outputs, including the
  # enable_thinking template kwarg.
  inferx_add_test(
    NAME chat_template_test
    SRCS chat_template_test.cc
    DEPS inferx::tokenizer nlohmann_json::nlohmann_json
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
    LABELS unit
  )
endif()
