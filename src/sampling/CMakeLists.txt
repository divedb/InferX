# inferx_sampling -- request-level sampling configuration, the batched
# runtime metadata, and the sampling pipeline (greedy fast path today; the
# fused bias/penalty/filter/sample kernels land in sampling_kernels.cu).
# Collective implementations depend on ops, and are usable by the runner
# without pulling in the worker-process engine layer.
inferx_add_library(
  NAME sampling
  SRCS
    sampling_params.cc
    sampler.cc
    sampling_kernels.cu
  DEPS inferx::ops absl::status absl::span
)
