if(INFERX_ENABLE_TOKENIZERS)
  inferx_add_test(
    NAME chat_template_test
    SRCS chat_template_test.cc
    DEPS inferx::http
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
    LABELS integration
  )
  # Tokenizer facade tests: load/encode/decode parity with the fixture,
  # option overloads, batching, and metadata.
  inferx_add_test(
    NAME tokenizer_test
    SRCS tokenizer_test.cc
    DEPS inferx::prompt_service
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
    LABELS integration
  )

  # Integration tests for the prompt-processor pool: token-for-token parity
  # with the facade, pre-tokenized input validation, bounded admission,
  # cancellation before/during prepare, request timeouts, and cooperative
  # drain. Determinism comes from holding a worker with a slow encode (a
  # large many-word text) and polling Health for observed queue/busy state
  # before issuing the action under test.
  inferx_add_test(
    NAME prompt_processor_pool_test
    SRCS prompt_processor_pool_test.cc
    DEPS inferx::prompt_service
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
    TIMEOUT 120
    LABELS integration
  )

  # Incremental detokenizer tests: suffix-only output (concatenated deltas
  # equal the full decode), UTF-8 holdback across append boundaries, window
  # sliding over long generations, flush on finish, and cancel.
  inferx_add_test(
    NAME detokenizer_test
    SRCS detokenizer_test.cc
    DEPS inferx::prompt_service
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
    LABELS integration
  )

  # End-to-end HTTP API tests for the vLLM-aligned surface: real server on
  # an ephemeral loopback port with a scripted fake engine (no GPU/engine
  # link). Covers health/models routes, auth, completions unary and SSE
  # bytes, error frames after headers, validation envelopes, and the
  # disconnect->abort path.
  inferx_add_test(
    NAME http_api_test
    SRCS http_api_test.cc
    DEPS inferx::http
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
    TIMEOUT 180
    LABELS integration
  )
endif()
