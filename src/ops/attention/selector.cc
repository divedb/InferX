#include "inferx/ops/attention/registry.h"

#include <algorithm>

#include "inferx/ops/attention/backends/flash_dense.h"
#include "inferx/ops/attention/backends/reference.h"

namespace inferx::attn {

void RegisterBuiltinBackends(BackendRegistry& registry) {
  registry.Add<FlashDenseBackend>(/*priority=*/100);
  // The reference backend is a correctness oracle and bring-up tool, never
  // a silent production fallback (docs/attention.md §7.3).
  registry.Add<ReferenceBackend>(/*priority=*/0);
}

BackendSelector::BackendSelector(const BackendRegistry& registry, Env env)
    : registry_(registry), env_(env) {
  ordered_.reserve(registry_.Entries().size());
  for (const BackendRegistry::Entry& entry : registry_.Entries()) {
    ordered_.push_back(&entry);
  }
  const auto by_priority = [](const BackendRegistry::Entry* a,
                                const BackendRegistry::Entry* b) {
    if (a->priority != b->priority) return a->priority > b->priority;
    // Stable for equal priorities: registration order.
    return false;
  };
  std::stable_sort(ordered_.begin(), ordered_.end(), by_priority);
}

Selection BackendSelector::Select(const AttentionSpec& spec) const {
  Selection selection;

  const auto try_backend = [&](const Backend& backend) -> bool {
    const Verdict verdict = CheckCapability(backend.GetCapability(), spec, env_);
    if (verdict) {
      selection.backend = &backend;
      return true;
    }
    selection.rejected.emplace_back(backend.GetCapability().name, verdict);
    return false;
  };

  if (!env_.forced_backend.empty()) {
    for (const BackendRegistry::Entry* entry : ordered_) {
      if (entry->backend->GetCapability().name == env_.forced_backend) {
        try_backend(*entry->backend);
        return selection;  // Forced: failure is an error, never a fallback.
      }
    }
    selection.rejected.emplace_back(
        env_.forced_backend, Verdict{Reject::kMechanism, "no backend with this name is registered"});
    return selection;
  }

  for (const BackendRegistry::Entry* entry : ordered_) {
    if (entry->backend->GetCapability().selectable_only_if_forced) continue;
    if (try_backend(*entry->backend)) return selection;
  }
  return selection;
}

}  // namespace inferx::attn
