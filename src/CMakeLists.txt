add_subdirectory(core)

# inferx_cache -- the paged KV block pool: host-side block bookkeeping for the
# scheduler over device-side KV storage.
add_library(inferx_cache
  cache/kv_block_pool.cc
)
add_library(inferx::cache ALIAS inferx_cache)
target_link_libraries(inferx_cache PUBLIC inferx::core)
target_compile_options(inferx_cache PRIVATE -Wall -Wextra -Wpedantic)

# inferx_ops -- the portable compute-op layer. Validation and dispatch are
# ordinary C++; each backend directory (ops/cpu, ops/cuda) is compiled
# separately and keeps its library-specific details to itself. The CPU
# backend's Highway kernels are compiled once per SIMD target and picked at
# runtime; the CUDA backend alone sees FlashInfer and nvcc.
add_library(inferx_ops
  ops/rms_norm.cc
  ops/cpu/rms_norm.cc
  ops/cuda/rms_norm.cu
)
add_library(inferx::ops ALIAS inferx_ops)
target_link_libraries(inferx_ops PUBLIC inferx::core
                      PRIVATE CUDA::cudart hwy)
target_include_directories(inferx_ops PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  ${PROJECT_SOURCE_DIR}/third_party/flashinfer/include
  ${PROJECT_SOURCE_DIR}/third_party/flashinfer/3rdparty/cccl/libcudacxx/include
  ${PROJECT_SOURCE_DIR}/third_party/flashinfer/3rdparty/cutlass/include)
target_compile_options(inferx_ops PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic>)

# inferx_models -- per-architecture model implementations over ops. Shared
# checkpoint machinery (config parsing, safetensors, weight upload) lives at
# the top level; each architecture directory owns its weights and forward
# flow and registers in Model::Load only.
add_library(inferx_models
  models/model.cc
  models/model_config.cc
  models/safetensors.cc
  models/weight_upload.cc
  models/llama/llama.cc
  models/qwen3/qwen3.cc
)
add_library(inferx::models ALIAS inferx_models)
target_link_libraries(inferx_models
  PUBLIC inferx::core inferx::cache inferx::ops
  PRIVATE nlohmann_json::nlohmann_json absl::flat_hash_map absl::span
          absl::status absl::statusor absl::strings)
target_include_directories(inferx_models PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_options(inferx_models PRIVATE -Wall -Wextra -Wpedantic)

# Architecture, scheduling, and orchestration are ordinary C++.
add_library(inferx_engine
  engine/request.cc
  engine/request_queue.cc
  engine/scheduler.cc
  models/model_runner.cc
  tokenizer/tokenizer.cc
)
add_library(inferx::engine ALIAS inferx_engine)
target_link_libraries(inferx_engine
  PUBLIC inferx::core inferx::cache inferx::models
         absl::flat_hash_map absl::status absl::statusor absl::strings)
if(INFERX_ENABLE_TOKENIZERS)
  target_link_libraries(inferx_engine PUBLIC inferx_tokenizers)
endif()
target_compile_options(inferx_engine PRIVATE -Wall -Wextra -Wpedantic)
