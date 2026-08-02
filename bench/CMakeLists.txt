# Benchmarks. Built only above the CUDA floor -- every one of them measures
# device work, so there is nothing here a host-only build could run.
#
# Benchmarks are not tests and are never run by ctest: they need a quiet machine
# and ideally locked clocks, and a number that regresses under CI noise would be
# a false alarm rather than a signal.

function(inferx_add_bench name)
  add_executable(${name} ${ARGN})
  target_link_libraries(${name} PRIVATE
    inferx::core
    inferx::common
    inferx::model
    inferx::kernels
    inferx::scheduler
    inferx::flags
  )
  # So that #include "bench/cuda_timer.h" resolves the same way our public
  # headers do, rather than by relative path.
  target_include_directories(${name} PRIVATE ${PROJECT_SOURCE_DIR})
  set_target_properties(${name} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/bench")
endfunction()

inferx_add_bench(gemm_bench gemm_bench.cc)
inferx_add_bench(decode_step_bench decode_step_bench.cc)
inferx_add_bench(decode_breakdown_bench decode_breakdown_bench.cc)
inferx_add_bench(prefill_bench prefill_bench.cc)

# The two that drive the scheduler rather than a hand-built batch. Inter-token
# latency and prefix reuse are both properties of how steps are composed, so
# there is nothing to measure without the component that composes them.
inferx_add_bench(tbt_bench tbt_bench.cc)
inferx_add_bench(prefix_cache_bench prefix_cache_bench.cc)
inferx_add_bench(eviction_bench eviction_bench.cc)

# Needs FlashInfer to have something to compare against.
if(INFERX_HAVE_FLASHINFER)
  inferx_add_bench(attention_bench attention_bench.cc)
endif()
