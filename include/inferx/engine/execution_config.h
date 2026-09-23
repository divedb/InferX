/// \file
/// \brief Execution strategy selection (vLLM CompilationConfig analogue).

#ifndef INFERX_ENGINE_EXECUTION_CONFIG_H_
#define INFERX_ENGINE_EXECUTION_CONFIG_H_

#include <string>
#include <vector>

namespace inferx {

/// \brief How the model executes: CUDA graph replay and the attention
///        implementation.
struct ExecutionConfig {
  /// \brief Replay pure-decode shapes when the model guarantees stable
  ///        storage (vLLM analog: --enforce-eager, inverted).
  bool enable_cuda_graphs = false;
  /// \brief Decode batch sizes to capture CUDA graphs for; empty captures
  ///        an automatically chosen size set.
  std::vector<int> cudagraph_capture_sizes;
  /// \brief Largest decode batch size to capture; 0 chooses automatically.
  int max_cudagraph_capture_size = 0;
  /// \brief Attention implementation: flashinfer; default/flash are
  ///        aliases.
  std::string attention_backend = "flashinfer";
};

}  // namespace inferx

#endif  // INFERX_ENGINE_EXECUTION_CONFIG_H_
