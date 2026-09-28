/// \file
/// \brief Execution strategy selection (vLLM CompilationConfig analogue).

#ifndef INFERX_CONFIG_EXECUTION_CONFIG_H_
#define INFERX_CONFIG_EXECUTION_CONFIG_H_

#include <string>
#include <vector>

namespace inferx {

/// \brief How the model executes: CUDA graph replay and the attention
///        implementation.
struct ExecutionConfig {
  /// Replay pure-decode shapes when the model guarantees stable storage
  /// (vLLM analog: --enforce-eager, inverted).
  /// EXAMPLE: --cuda-graphs
  bool enable_cuda_graphs = false;
  /// Decode batch sizes to capture CUDA graphs for; empty captures an
  /// automatically chosen size set.
  /// EXAMPLE: --cudagraph-capture-sizes 1,2,4,8
  std::vector<int> cudagraph_capture_sizes;
  /// Largest decode batch size to capture; 0 chooses automatically.
  /// EXAMPLE: --max-cudagraph-capture-size 64
  int max_cudagraph_capture_size = 0;
};

}  // namespace inferx

#endif  // INFERX_CONFIG_EXECUTION_CONFIG_H_
