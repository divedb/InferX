#ifndef INFERX_ATTN_REGISTRY_H_
#define INFERX_ATTN_REGISTRY_H_

#include <memory>
#include <span>
#include <utility>
#include <vector>

#include "inferx/ops/attention/backend.h"
#include "inferx/ops/attention/capability.h"

namespace inferx::attn {

class BackendRegistry {
 public:
  struct Entry {
    std::unique_ptr<Backend> backend;
    int priority;
  };

  template <BackendLike B>
  void Add(int priority = 0) {
    backends_.push_back({std::make_unique<B>(), priority});
  }
  [[nodiscard]] std::span<const Entry> Entries() const noexcept { return backends_; }

 private:
  std::vector<Entry> backends_;
};

/// \brief Explicit registration function in a single .cc: no
///        static-initialization-order tricks. Adds the reference backend
///        (selectable only when forced) and the FlashInfer-backed dense
///        backend.
void RegisterBuiltinBackends(BackendRegistry& registry);

/// \brief Selection outcome; `rejected` carries the full rejection trail.
struct Selection {
  const Backend* backend = nullptr;
  std::vector<std::pair<std::string_view, Verdict>> rejected;
  [[nodiscard]] bool Ok() const { return backend != nullptr; }
};

/// \brief Filters the registry by Capability and ranks by priority.
///        Deterministic (stable sort), so multi-rank runs choose
///        identically.
class BackendSelector {
 public:
  BackendSelector(const BackendRegistry& registry, Env env);

  /// \brief Never throws: an empty Selection (with the trail) means
  ///        nothing fits and the caller formats the error. A forced backend
  ///        that cannot serve the spec is an error, never a silent
  ///        fallback.
  [[nodiscard]] Selection Select(const AttentionSpec& spec) const;

 private:
  const BackendRegistry& registry_;
  Env env_;
  /// Stable priority order, resolved once.
  std::vector<const BackendRegistry::Entry*> ordered_;
};

}  // namespace inferx::attn

#endif  // INFERX_ATTN_REGISTRY_H_
