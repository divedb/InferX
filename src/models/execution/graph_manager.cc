#include "inferx/models/execution/graph_manager.h"

#include <map>
#include <utility>

namespace inferx::models::execution {

StatusOr<Tensor> GraphManager::Execute(int key, bool eligible, uint64_t workspace_generation,
                                       Stream stream, const ModelLaunch& launch) {
  if (generation_ != workspace_generation) {
    // Workspace was reallocated under us: every captured address is stale.
    DestroyAll();
    generation_ = workspace_generation;
  }
  if (!eligible) {
    graphs_.erase(key);
    return launch();
  }

  Entry& graph = graphs_[key];
  if (!graph.warmed) {
    // Real execution whose result this step uses; libraries initialize
    // against the exact shape before any capture.
    INFERX_ASSIGN_OR_RETURN(Tensor logits, launch());
    graph.warmed = true;
    return logits;
  }
  if (graph.exec.handle == nullptr) {
    // Capture records the callable without executing it, so state does not
    // advance here; the replay below performs this step's real work.
    INFERX_RETURN_IF_ERROR(runtime_.BeginCapture(stream));
    auto captured = launch();
    auto instantiated = runtime_.EndCaptureAndInstantiate(stream);
    if (!captured.ok() || !instantiated.ok()) {
      if (instantiated.ok() && instantiated->handle != nullptr) {
        (void)runtime_.DestroyGraph(*instantiated);
      }
      graphs_.erase(key);
      return captured.ok() ? instantiated.status() : captured.status();
    }
    graph.exec = std::move(*instantiated);
    graph.logits = std::move(captured).value();
  }
  INFERX_RETURN_IF_ERROR(runtime_.LaunchGraph(graph.exec, stream));
  return *graph.logits;
}

void GraphManager::DestroyAll() {
  for (auto& [key, graph] : graphs_) {
    if (graph.exec.handle != nullptr) (void)runtime_.DestroyGraph(graph.exec);
  }
  graphs_.clear();
}

}  // namespace inferx::models::execution
