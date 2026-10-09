if(INFERX_ENABLE_TOKENIZERS)
  # Integration tests for the prompt-preparation worker pool (token-for-token
  # parity with the facade, admission bounds, cancellation, timeouts, drain).
  inferx_add_test(
    NAME tokenizer_pool_test
    SRCS tokenizer_pool_test.cc
    DEPS inferx::server
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
    TIMEOUT 120
    LABELS integration
  )
endif()
