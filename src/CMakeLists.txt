add_subdirectory(core)

# inferx_cache -- the paged KV block pool: host-side block bookkeeping for the
# scheduler over device-side KV storage.
add_library(inferx_cache
  cache/kv_block_pool.cc
  cache/recurrent_state_pool.cc
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
  ops/gather.cc
  ops/linear.cc
  ops/elementwise.cc
  ops/rotary.cc
  ops/attention.cc
  ops/moe.cc
  ops/mla.cc
  ops/gdn.cc
  ops/flash_attention.cc
  ops/cuda/flash_attention.cu
  ops/cuda/generic_attention.cu
  ops/cuda/moe.cu
  ops/cuda/mla.cu
  ops/cuda/gdn.cu
  ops/cuda/sampling.cu
  ops/cuda/gather.cu
  ops/cuda/linear.cu
  ops/cuda/elementwise.cu
  ops/cuda/rotary.cu
  ops/cuda/attention.cu
  ops/cuda/cublas.cu
)
set_source_files_properties(ops/cuda/flash_attention.cu PROPERTIES
  COMPILE_OPTIONS "--expt-relaxed-constexpr")
# Highway re-includes rms_norm.cc once per SIMD target through
# HWY_TARGET_INCLUDE, which resolves against the include path.
set_source_files_properties(ops/cpu/rms_norm.cc PROPERTIES
  COMPILE_OPTIONS "-I${CMAKE_CURRENT_SOURCE_DIR}")
add_library(inferx::ops ALIAS inferx_ops)
target_link_libraries(inferx_ops PUBLIC inferx::core
                      PRIVATE CUDA::cudart CUDA::cublas hwy)
target_include_directories(inferx_ops PRIVATE
  ${PROJECT_SOURCE_DIR}/third_party/flashinfer/include
  ${PROJECT_SOURCE_DIR}/third_party/flashinfer/3rdparty/cccl/libcudacxx/include
  ${PROJECT_SOURCE_DIR}/third_party/flashinfer/3rdparty/cutlass/include)
target_compile_options(inferx_ops PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic>)

# inferx_sampling -- request-level sampling configuration, the batched
# runtime metadata, and the sampling pipeline (greedy fast path today; the
# fused bias/penalty/filter/sample kernels land in sampling/sampling_kernels.cu).
add_library(inferx_sampling
  sampling/sampling_params.cc
  sampling/sampler.cc
)
add_library(inferx::sampling ALIAS inferx_sampling)
target_link_libraries(inferx_sampling PUBLIC inferx::ops absl::status absl::span)
target_compile_options(inferx_sampling PRIVATE -Wall -Wextra -Wpedantic)

# inferx_models -- reusable transformer components under models/components
# (config + weights + execution), typed decoder composition under models/causal,
# checkpoint mapping under models/loading, and family factories selected by
# checkpoint identity. Explicit factory references retain families in static links.
add_library(inferx_models
  models/model.cc
  models/model_registry.cc
  models/families/llama.cc
  models/families/qwen3.cc
  models/families/dense_families.cc
  models/families/moe_families.cc
  models/causal/config_parser.cc
  models/checkpoint_config.cc
  models/checkpoint.cc
  models/safe_tensors_reader.cc
  models/safe_tensors_view.cc
  models/components/attention.cc
  models/components/decoder_layer.cc
  models/components/mlp.cc
  models/components/moe.cc
  models/components/mla.cc
  models/components/gdn.cc
  models/components/parallel_linear.cc
  models/components/qkv_linear.cc
  models/causal/causal_lm.cc
  models/causal/decoder_stack.cc
  models/causal/decoder_config.cc
  models/loading/weight_loader.cc
)
add_library(inferx::models ALIAS inferx_models)
target_link_libraries(inferx_models
  PUBLIC inferx::core inferx::cache inferx::ops inferx::boost
         nlohmann_json::nlohmann_json
  PRIVATE absl::flat_hash_map absl::span
          absl::status absl::statusor absl::strings)
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
  PUBLIC inferx::core inferx::cache inferx::models inferx::sampling
         absl::flat_hash_map absl::status absl::statusor absl::strings)
if(INFERX_ENABLE_TOKENIZERS)
  target_link_libraries(inferx_engine PUBLIC inferx::tokenizers)
endif()
target_compile_options(inferx_engine PRIVATE -Wall -Wextra -Wpedantic)

# inferx_dist -- tensor-parallel worker topology: the controller-side
# process pool (posix_spawn, one worker per device), the IPC wire protocol
# over Boost.InterProcess message queues, and the worker-side engine loop.
# Collectives sit behind dist/comm.h; SingleRankComm (identity) is the only
# backend until NCCL lands with multi-GPU support.
add_library(inferx_dist
  dist/worker_ipc.cc
  dist/worker_launch.cc
  dist/worker_main.cc
)
add_library(inferx::dist ALIAS inferx_dist)
target_link_libraries(inferx_dist
  PUBLIC inferx::engine inferx::boost
          absl::status absl::statusor absl::strings)
target_compile_options(inferx_dist PRIVATE -Wall -Wextra -Wpedantic)

# inferx_bench -- CLI-independent benchmark entry points (`inferx bench *`).
# Business libraries never include anything from apps/.
add_library(inferx_bench
  bench/workload.cc
  bench/throughput.cc
  bench/latency.cc
)
add_library(inferx::bench ALIAS inferx_bench)
target_link_libraries(inferx_bench
  PUBLIC inferx::engine
  PRIVATE nlohmann_json::nlohmann_json CUDA::cudart)
target_compile_options(inferx_bench PRIVATE -Wall -Wextra -Wpedantic)

# inferx_diagnostic -- numerical debugging tools (`inferx diagnostic *`).
add_library(inferx_diagnostic diagnostic/replay_logits.cc)
add_library(inferx::diagnostic ALIAS inferx_diagnostic)
target_link_libraries(inferx_diagnostic
  PUBLIC inferx::engine
  PRIVATE nlohmann_json::nlohmann_json)
target_compile_options(inferx_diagnostic PRIVATE -Wall -Wextra -Wpedantic)

# inferx_server -- OpenAI-compatible HTTP server (`inferx serve`). The
# gateway owns the engine thread; the coroutine front end runs on
# Boost.Beast/Asio (interface targets from InferXDependencies.cmake); the
# TokenizerPool runs the prompt-preparation worker threads.
add_library(inferx_server
  server/serve.cc
  server/api.cc
  server/engine_gateway.cc
  server/http_server.cc
  server/tokenizer_pool.cc
)
add_library(inferx::server ALIAS inferx_server)
target_link_libraries(inferx_server
  PUBLIC inferx::engine inferx::boost
  PRIVATE inferx::beast nlohmann_json::nlohmann_json
          absl::flat_hash_map absl::status absl::statusor)
target_compile_options(inferx_server PRIVATE -Wall -Wextra -Wpedantic)
