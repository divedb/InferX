# inferx_bench -- CLI-independent benchmark entry points (`inferx bench *`).
# Business libraries never include anything from apps/.
inferx_add_library(
  NAME bench
  SRCS
    workload.cc
    throughput.cc
    latency.cc
  DEPS inferx::engine
  PRIVATE_DEPS nlohmann_json::nlohmann_json CUDA::cudart
)
