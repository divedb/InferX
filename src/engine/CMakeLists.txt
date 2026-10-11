# inferx_engine -- architecture, scheduling, and orchestration are ordinary
# C++. The tokenizer front end (tokenizer/tokenizer.cc) belongs to this
# layer; the model layer ships its own library with the runner.
inferx_add_library(
  NAME engine
  SRCS
    request.cc
    request_queue.cc
    scheduler.cc
    model_executor.cc
    ../tokenizer/tokenizer.cc
  PRIVATE_INCLUDE_DIRS ${PROJECT_SOURCE_DIR}/src
  DEPS
    inferx::core
    inferx::cache
    inferx::models
    inferx::sampling
    inferx::comm
    absl::flat_hash_map
    absl::status
    absl::statusor
    absl::strings
)

if(INFERX_ENABLE_TOKENIZERS)
  target_link_libraries(inferx_engine PUBLIC inferx::tokenizers)
endif()
