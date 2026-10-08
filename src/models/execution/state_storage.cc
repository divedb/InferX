#include "inferx/models/execution/state_storage.h"

namespace inferx::models::execution {

Status StateStorage::Init(const std::vector<LayerStateSpec>& specs, int num_kv_blocks,
                          int block_size, int max_slots, DeviceId device) {
  // One heterogeneous paged group covers every paged layer; recurrent
  // layers get their slot pool. Bindings are assigned in declaration order.
  std::vector<KvLayout> layouts;
  std::vector<RecurrentStateSpec> recurrent_specs;
  for (const auto& spec : specs) {
    if (const auto* paged = std::get_if<PagedKvStateSpec>(&spec)) {
      layouts.push_back(paged->layout);
    } else {
      recurrent_specs.push_back(std::get<RecurrentStateSpec>(spec));
    }
  }
  INFERX_ASSIGN_OR_RETURN(auto kv,
                          KvBlockPool::Create(num_kv_blocks, block_size, layouts, device));
  kv_ = std::make_unique<KvBlockPool>(std::move(kv));
  if (!recurrent_specs.empty()) {
    INFERX_ASSIGN_OR_RETURN(auto recurrent,
                            RecurrentStatePool::Create(recurrent_specs, max_slots, device));
    recurrent_ = std::make_unique<RecurrentStatePool>(std::move(recurrent));
  }

  bindings_.paged_kv = kv_.get();
  bindings_.recurrent = recurrent_.get();
  int64_t next_paged = 0, next_recurrent = 0;
  for (const auto& spec : specs) {
    if (std::holds_alternative<PagedKvStateSpec>(spec)) {
      bindings_.layers.push_back(PagedKvState{next_paged++});
    } else {
      bindings_.layers.push_back(RecurrentState{next_recurrent++});
    }
  }
  return OkStatus();
}

}  // namespace inferx::models::execution
