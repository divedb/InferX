#include "inferx/models/execution/request_state.h"

namespace inferx::models::execution {

RequestStateTable::RequestStateTable(int max_slots) : max_slots_(max_slots) {
  slots_.resize(static_cast<size_t>(max_slots));
  for (size_t slot = slots_.size(); slot > 0; --slot) {
    free_slots_.push_back(slot - 1);
  }
}

StatusOr<RequestRecord*> RequestStateTable::Admit(const NewRequestData& data) {
  if (index_.contains(data.request_id)) {
    return InvalidArgumentError("duplicate runner request ", data.request_id);
  }
  if (free_slots_.empty()) {
    return ResourceExhaustedError("request slots are exhausted");
  }
  const size_t slot = free_slots_.back();
  free_slots_.pop_back();
  RequestRecord& record = slots_[slot].emplace();
  record.request_id = data.request_id;
  record.generation += 1;
  record.prompt.assign(data.prompt_token_ids.begin(), data.prompt_token_ids.end());
  record.params = data.sampling_params;
  record.block_ids = data.block_ids;
  record.committed = data.num_computed_tokens;
  record.generated = 0;
  record.last_sampled = -1;
  record.recurrent_slot = -1;
  index_.emplace(data.request_id, slot);
  return &record;
}

RequestRecord* RequestStateTable::Find(RequestId id) {
  const auto it = index_.find(id);
  if (it == index_.end()) return nullptr;
  return &*slots_[it->second];
}

Status RequestStateTable::Retire(RequestId id) {
  const auto it = index_.find(id);
  if (it == index_.end()) return InvalidArgumentError("unknown finished request ", id);
  RequestRecord& record = *slots_[it->second];
  if (record.recurrent_slot >= 0) ReleaseRecurrentSlot(record.recurrent_slot);
  slots_[it->second].reset();
  free_slots_.push_back(it->second);
  index_.erase(it);
  return OkStatus();
}

void RequestStateTable::InitRecurrentSlots(int slots) {
  free_recurrent_.clear();
  free_recurrent_.reserve(slots);
  for (int slot = slots; slot > 0; --slot) free_recurrent_.push_back(slot - 1);
}

StatusOr<int> RequestStateTable::AcquireRecurrentSlot() {
  if (free_recurrent_.empty()) {
    return ResourceExhaustedError("recurrent state slots are exhausted");
  }
  const int slot = free_recurrent_.back();
  free_recurrent_.pop_back();
  return slot;
}

void RequestStateTable::ReleaseRecurrentSlot(int slot) {
  if (slot >= 0) free_recurrent_.push_back(slot);
}

}  // namespace inferx::models::execution
