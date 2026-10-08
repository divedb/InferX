/// \file
/// \brief Runner-private decode-graph manager. Captures the same prepared
///        model callable eager execution uses -- Forward plus
///        ComputeLogits, nothing else. Sampling stays eager: its parameter
///        uploads and greedy/stochastic selection are host decisions that
///        graph replay would freeze at stale values.
///
/// Graph key today is the exact decode batch size; dtype, cache layout and
/// topology are fixed for the manager's lifetime and covered by the
/// workspace generation. Padding buckets are deferred: exact-size graphs
/// need no active-count masking.

#ifndef INFERX_MODELS_EXECUTION_GRAPH_MANAGER_H_
#define INFERX_MODELS_EXECUTION_GRAPH_MANAGER_H_

#include <functional>
#include <map>
#include <optional>

#include "inferx/core/device_runtime.h"
#include "inferx/core/status.h"
#include "inferx/core/tensor.h"

namespace inferx::models::execution {

class GraphManager {
 public:
  /// \brief Enqueues Forward + ComputeLogits and returns the logits view.
  ///        The view must borrow stable model workspace so a captured
  ///        address stays valid across replays.
  using ModelLaunch = std::function<StatusOr<Tensor>()>;

  explicit GraphManager(DeviceRuntime& runtime) : runtime_(runtime) {}

  /// \brief Runs one launch, through a captured graph when eligible.
  ///
  /// Sequence per graph key: the first occurrence warms up eagerly (real
  /// execution whose result is used, letting libraries initialize); the
  /// second captures the callable without executing it and immediately
  /// replays; later occurrences replay directly. Capture itself executes
  /// nothing, so warming and capturing never double-advance state.
  ///
  /// \param key            Graph key (exact decode batch size).
  /// \param eligible       Runner-side eligibility (uniform decode, CUDA,
  ///                       graphs enabled) AND the model's PrepareStep
  ///                       eligibility.
  /// \param workspace_generation  Captured graphs become invalid when the
  ///                       model's workspace is reallocated.
  /// \param launch         The prepared model callable.
  /// \return               The logits for this step (eager result, or the
  ///                       stable view filled by replay).
  StatusOr<Tensor> Execute(int key, bool eligible, uint64_t workspace_generation,
                           Stream stream, const ModelLaunch& launch);

  /// \brief Destroys every graph. The runner calls this before any buffer
  ///        a graph references is freed.
  void DestroyAll();

 private:
  struct Entry {
    GraphExec exec;
    std::optional<Tensor> logits;  ///< Stable view written by each replay.
    bool warmed = false;
  };

  DeviceRuntime& runtime_;
  std::map<int, Entry> graphs_;
  uint64_t generation_ = 0;
};

}  // namespace inferx::models::execution

#endif  // INFERX_MODELS_EXECUTION_GRAPH_MANAGER_H_
