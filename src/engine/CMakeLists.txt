# inferx_engine -- architecture, scheduling, and orchestration are ordinary
# C++. ModelRunner, Worker, persistent InputBatch state, and the request/
# queue types live here. Tokenizer-facing serving components live in the
# prompt_service library under src/server (docs/tokenizer_design.md).
inferx_add_library(
  NAME engine
  SRCS
    request.cc
    request_queue.cc
    scheduler.cc
    model_runner.cc
    worker.cc
    input_batch.cc
  PRIVATE_INCLUDE_DIRS ${PROJECT_SOURCE_DIR}/src
  DEPS
    inferx::core
    inferx::cache_manager
    inferx::models
    inferx::sampling
    inferx::comm
    absl::flat_hash_map
    absl::status
    absl::statusor
    absl::strings
)
