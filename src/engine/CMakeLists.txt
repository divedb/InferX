# inferx_engine -- architecture, scheduling, and orchestration are ordinary
# C++. The runner (models/model_runner.cc) and tokenizer front end
# (tokenizer/tokenizer.cc) belong to this layer, so they are listed relative
# to their owning directories.
inferx_add_library(
  NAME engine
  SRCS
    request.cc
    request_queue.cc
    scheduler.cc
    ../models/model_runner.cc
    ../tokenizer/tokenizer.cc
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
