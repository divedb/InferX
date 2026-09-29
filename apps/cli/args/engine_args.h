#pragma once

#include <CLI/CLI.hpp>
#include <set>
#include <string>
#include <vector>

#include "cli/app.h"
#include "inferx/config/cache_config.h"
#include "inferx/config/execution_config.h"
#include "inferx/config/parallel_config.h"
#include "inferx/config/scheduler_config.h"

namespace inferx::cli {

/// \brief vLLM CacheConfig/SchedulerConfig/CompilationConfig analogue:
/// execution capacity. Checkpoint identity and presentation live in
/// ModelConfigArgs.
struct EngineArgs {
  int max_num_seqs = 32;                // vLLM: --max-num-seqs
  int max_num_batched_tokens = 4096;    // vLLM: --max-num-batched-tokens
  int64_t num_kv_blocks = 2048;         // InferX: explicit pool size (vLLM derives it
                                        //             from --gpu-memory-utilization)
  int64_t block_size = 16;              // vLLM: --block-size
  bool cuda_graphs = false;             // vLLM analog: --enforce-eager, inverted
  int64_t kv_cache_memory_bytes = 0;    // vLLM: --kv-cache-memory-bytes; 0
                                        //             derives from num_kv_blocks
  bool chunked_prefill = true;          // vLLM: --enable-chunked-prefill
  std::vector<int> cudagraph_capture_sizes;   // vLLM: --cudagraph-capture-sizes
  int max_cudagraph_capture_size = 0;         // vLLM: --max-cudagraph-capture-size
  int tensor_parallel_size = 1;               // vLLM: --tensor-parallel-size; one
                                              // rank per GPU when > 1

  /// Binds this group's options to `sub`. The struct instance must outlive
  /// the App: commands keep it inside their args struct, which the callback
  /// owns via shared_ptr.
  void AddOptions(CLI::App& sub) {
    CLI::Option_group* g = sub.add_option_group("Engine", "execution capacity");
    // Deep-tuning knobs live in the "Advanced" group: excluded from normal
    // help, shown by --help=all. Everything here keeps parsing identically.
    g->add_option("--max-num-seqs", max_num_seqs, "Maximum number of sequences per iteration")
        ->type_name("N")
        ->capture_default_str()
        ->check(CLI::PositiveNumber);
    g->add_option("--max-num-batched-tokens", max_num_batched_tokens,
                  "Maximum number of tokens per batch")
        ->type_name("N")
        ->capture_default_str()
        ->check(CLI::Range(16, 1 << 20));
    g->add_option("--num-kv-blocks", num_kv_blocks,
                  "InferX: total KV cache blocks preallocated across all layers")
        ->type_name("N")
        ->group("Advanced")
        ->capture_default_str()
        ->check(CLI::PositiveNumber);
    g->add_option("--block-size", block_size, "Token capacity of one KV cache block")
        ->type_name("N")
        ->group("Advanced")
        ->capture_default_str()
        ->check(CLI::IsMember({int64_t{16}, int64_t{32}, int64_t{64}}));
    g->add_flag("--cuda-graphs", cuda_graphs,
                "Capture and replay decode CUDA graphs (vLLM: --enforce-eager, inverted)")
        ->group("Advanced");
    g->add_option("--kv-cache-memory-bytes", kv_cache_memory_bytes,
                  "Exact KV cache size in bytes; 0 derives it from "
                  "--num-kv-blocks")
        ->type_name("BYTES")
        ->capture_default_str()
        ->check(CLI::Range(int64_t{0}, int64_t{1} << 40));
    CLI::Option* chunked = g->add_flag("--enable-chunked-prefill", chunked_prefill,
                                       "Allow splitting long prompts across scheduler steps "
                                       "(default)");
    chunked->group("Advanced");
    CLI::Option* whole = g->add_flag(
        "--no-enable-chunked-prefill", [this](std::int64_t) { chunked_prefill = false; },
        "Require whole prompts to prefill in one step");
    whole->excludes(chunked)->group("Advanced");
    g->add_option("--cudagraph-capture-sizes", cudagraph_capture_sizes,
                  "Decode batch sizes to capture graphs for; empty selects "
                  "automatically")
        ->type_name("SIZES")
        ->group("Advanced")
        ->delimiter(',')
        ->check(CLI::Range(1, 4096));
    g->add_option("--max-cudagraph-capture-size", max_cudagraph_capture_size,
                  "Largest decode batch size to capture; 0 selects "
                  "automatically")
        ->type_name("N")
        ->group("Advanced")
        ->capture_default_str()
        ->check(CLI::Range(0, 4096));
    g->add_option("--tensor-parallel-size", tensor_parallel_size,
                  "Tensor-parallel GPU ranks, one per device id; requires NCCL and eager execution")
        ->type_name("N")
        ->group("Advanced")
        ->capture_default_str()
        ->check(CLI::Range(1, 255));
  }

  /// \brief KV cache pool sizing (vLLM CacheConfig analogue).
  CacheConfig BuildCacheConfig() const {
    CacheConfig cfg;
    cfg.num_kv_blocks = num_kv_blocks;
    cfg.block_size = block_size;
    cfg.kv_cache_memory_bytes = kv_cache_memory_bytes;
    return cfg;
  }

  /// \brief Execution strategy: CUDA graphs (vLLM CompilationConfig
  ///        analogue).
  ExecutionConfig BuildExecutionConfig() const {
    ExecutionConfig cfg;
    cfg.enable_cuda_graphs = cuda_graphs;
    cfg.cudagraph_capture_sizes = cudagraph_capture_sizes;
    cfg.max_cudagraph_capture_size = max_cudagraph_capture_size;
    return cfg;
  }

  /// \brief Scheduling policy knobs.
  SchedulerConfig BuildSchedulerConfig() const {
    SchedulerConfig cfg;
    cfg.max_num_seqs = max_num_seqs;
    cfg.max_num_batched_tokens = max_num_batched_tokens;
    cfg.chunked_prefill = chunked_prefill;
    return cfg;
  }

  /// \brief The controller's view of the tensor-parallel topology (vLLM
  ///        --tensor-parallel-size; rank 0). The runner assigns each GPU its rank.
  ParallelConfig BuildParallelConfig() const {
    ParallelConfig cfg;
    cfg.tensor_parallel_size = tensor_parallel_size;
    return cfg;
  }
};

}  // namespace inferx::cli
