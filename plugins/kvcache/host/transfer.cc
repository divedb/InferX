// Data plane of the host reference provider: memory import, load, store,
// and the transfer completion table. All copies are synchronous host
// memcpys, so accepted transfers are terminal when returned.

#include <cstring>
#include <new>
#include <utility>

#include "internal.h"

namespace kvc_host {
namespace {

bool StringIs(const KvcString& text, const char* literal) {
  const size_t length = std::strlen(literal);
  return text.size == length && (length == 0 || std::memcmp(text.data, literal, length) == 0);
}

uint64_t RowBytes(const Session::GroupCopy& group) {
  return group.payload_bytes / group.group.layer_count;
}

// Payload offset of (layer, component) within one block.
bool SlotPosition(const Session::GroupCopy& group, uint32_t layer, uint32_t component,
                  size_t* layer_at, size_t* component_at, uint64_t* offset) {
  *layer_at = group.layers.size();
  for (size_t i = 0; i < group.layers.size(); ++i)
    if (group.layers[i] == layer) *layer_at = i;
  if (*layer_at == group.layers.size()) return false;
  *component_at = group.components.size();
  for (size_t c = 0; c < group.components.size(); ++c)
    if (group.components[c].id == component) *component_at = c;
  if (*component_at == group.components.size()) return false;
  *offset = *layer_at * RowBytes(group) + group.slot_offsets[*component_at];
  return true;
}

}  // namespace

KvcStatus ImportMemory(KvcObject* object, const KvcMemoryImport* request, KvcObject** out) {
  return Guard([&]() -> KvcStatus {
    auto* session = reinterpret_cast<Session*>(object);
    if (!request || request->struct_size < sizeof(KvcMemoryImport) || !out)
      return ErrorStatus(KVC_INVALID_ARGUMENT, "import request is malformed");
    if (request->info.struct_size < sizeof(KvcMemoryInfo))
      return ErrorStatus(KVC_INVALID_ARGUMENT, "memory info record is too small");
    *out = nullptr;
    std::lock_guard<std::mutex> lock(session->mutex);
    if (session->closing) return ErrorStatus(KVC_SHUTTING_DOWN, "session is closing");
    if (!session->configured) return ErrorStatus(KVC_UNAVAILABLE, "session is not configured");

    const KvcMemoryInfo& info = request->info;
    if (info.locator_kind != KVC_LOCAL_ADDRESS)
      return ErrorStatus(KVC_UNSUPPORTED, "only local-address imports are supported");
    if (info.memory_type != KVC_HOST && info.memory_type != KVC_PINNED_HOST)
      return ErrorStatus(KVC_UNSUPPORTED, "only host memory can be imported");
    if (info.access != KVC_READ_ONLY && info.access != KVC_READ_WRITE)
      return ErrorStatus(KVC_INVALID_ARGUMENT, "access mode is invalid");
    if (!StringIs(info.device_runtime, "cpu") || !StringIs(info.device_identifier, "0"))
      return ErrorStatus(KVC_UNSUPPORTED, "import requires local CPU memory");
    if (info.byte_size == 0 || info.local_address == 0)
      return ErrorStatus(KVC_INVALID_ARGUMENT, "memory extent is invalid");
    if (!request->allocation_owner.retain || !request->allocation_owner.release)
      return ErrorStatus(KVC_INVALID_ARGUMENT, "allocation owner callbacks are required");

    auto* region = new (std::nothrow) Region();
    if (!region) return ErrorStatus(KVC_RESOURCE_EXHAUSTED, "region allocation failed");
    region->info = info;
    region->info.struct_size = sizeof(KvcMemoryInfo);
    region->info.external = KvcDescriptor{};
    region->runtime.assign(info.device_runtime.data, info.device_runtime.size);
    region->identifier.assign(info.device_identifier.data, info.device_identifier.size);
    region->info.device_runtime = {region->runtime.data(), region->runtime.size()};
    region->info.device_identifier = {region->identifier.data(), region->identifier.size()};
    region->owner_context = request->allocation_owner.context;
    region->owner_release = request->allocation_owner.release;
    region->header.kind = KVC_MEMORY;
    region->header.session = session;
    request->allocation_owner.retain(request->allocation_owner.context);
    session->ObjectCreated();
    *out = reinterpret_cast<KvcObject*>(region);
    return OkStatus();
  });
}

KvcStatus ResolveBindings(Session& session, const KvcSpan& span, const KvcBinding* bindings,
                          uint64_t binding_count, const std::vector<BlockPtr>& owned,
                          uint32_t required_access, bool writing,
                          std::vector<std::vector<bool>>* coverage, BindingPlan* out) {
  const Session::GroupCopy* group = session.FindGroup(span.group);
  if (!group) return ErrorStatus(KVC_INVALID_ARGUMENT, "span references an unknown group");
  if (!span.layers || span.layer_count == 0)
    return ErrorStatus(KVC_INVALID_ARGUMENT, "span selects no layers");
  if (!span.blocks || span.block_count == 0)
    return ErrorStatus(KVC_INVALID_ARGUMENT, "span selects no blocks");
  if (binding_count && !bindings)
    return ErrorStatus(KVC_INVALID_ARGUMENT, "bindings are missing");

  // Selected layers must be unique members of the group.
  std::vector<uint32_t> layers(span.layers, span.layers + span.layer_count);
  for (size_t i = 0; i < layers.size(); ++i) {
    bool member = false;
    for (uint32_t group_layer : group->layers)
      if (group_layer == layers[i]) member = true;
    if (!member) return ErrorStatus(KVC_INVALID_RANGE, "selected layer is not in the group");
    for (size_t j = 0; j < i; ++j)
      if (layers[j] == layers[i])
        return ErrorStatus(KVC_INVALID_ARGUMENT, "selected layers must be unique");
  }

  // Resolve span blocks against the pinned (load) or reserved (store) set.
  std::unordered_map<uint64_t, size_t> positions;  // Key -> position in owned.
  for (size_t i = 0; i < owned.size(); ++i)
    positions.emplace(Session::KeyId(owned[i]->record.key), i);
  std::unordered_map<uint64_t, size_t> selected;  // Key -> position in plan.
  out->group = group;
  out->blocks.clear();
  out->entries.clear();
  for (uint64_t b = 0; b < span.block_count; ++b) {
    const KvcSlice& slice = span.blocks[b];
    auto known = positions.find(Session::KeyId(slice.key));
    if (known == positions.end())
      return ErrorStatus(KVC_INVALID_RANGE, "span block is not in the handle manifest");
    const BlockPtr& block = owned[known->second];
    if (slice.first_token != 0 || slice.token_count != block->record.token_count)
      return ErrorStatus(KVC_UNSUPPORTED, "opaque components transfer whole blocks only");
    if (!selected.emplace(Session::KeyId(slice.key), out->blocks.size()).second)
      return ErrorStatus(KVC_INVALID_ARGUMENT, "span block slices overlap");
    out->blocks.push_back(block);
  }

  // Bindings must exactly cover (span blocks) x (layers) x (components).
  const uint64_t expected = span.block_count * layers.size() * group->group.component_count;
  if (binding_count != expected)
    return ErrorStatus(KVC_INVALID_ARGUMENT, "bindings do not exactly cover the selection");

  struct SlotKey {
    size_t block;  // Position in out->blocks.
    size_t owned;  // Position in owned / coverage.
    size_t bit;
    size_t layer_at;
    size_t component_at;
  };
  std::vector<SlotKey> slots;
  std::vector<const KvcBinding*> sources;
  std::vector<Region*> slot_regions;
  slots.reserve(binding_count);
  for (uint64_t i = 0; i < binding_count; ++i) {
    const KvcBinding& binding = bindings[i];
    if (binding.layout != KVC_OPAQUE_BYTES)
      return ErrorStatus(KVC_UNSUPPORTED, "only opaque-byte bindings are supported");
    auto slot = selected.find(Session::KeyId(binding.block.key));
    if (slot == selected.end())
      return ErrorStatus(KVC_INVALID_RANGE, "binding block is not in the span");
    const BlockPtr& block = out->blocks[slot->second];
    if (binding.block.first_token != 0 ||
        binding.block.token_count != block->record.token_count)
      return ErrorStatus(KVC_INVALID_RANGE, "binding extent differs from the span slice");
    bool layer_selected = false;
    for (uint32_t layer : layers) layer_selected = layer_selected || layer == binding.layer;
    if (!layer_selected) return ErrorStatus(KVC_INVALID_RANGE, "binding layer is not selected");

    size_t layer_at = 0, component_at = 0;
    uint64_t offset = 0;
    if (!SlotPosition(*group, binding.layer, binding.component, &layer_at, &component_at,
                      &offset))
      return ErrorStatus(KVC_INVALID_RANGE, "layer or component is not in the group");
    if (!Session::Owns(binding.region, KVC_MEMORY) ||
        Session::AsHeader(binding.region)->session != &session)
      return ErrorStatus(KVC_STALE_HANDLE, "binding region is invalid for this session");
    auto* region = reinterpret_cast<Region*>(binding.region);
    const uint64_t component_bytes = group->components[component_at].bytes_per_block;
    if (binding.byte_count != component_bytes)
      return ErrorStatus(KVC_INVALID_RANGE, "binding must cover the whole component");
    if (required_access != 0 && region->info.access != required_access)
      return ErrorStatus(KVC_INVALID_ARGUMENT, "region access mode is insufficient");
    if (binding.byte_offset > region->info.byte_size ||
        binding.byte_count > region->info.byte_size - binding.byte_offset)
      return ErrorStatus(KVC_INVALID_RANGE, "binding exceeds its region");

    SlotKey key{slot->second, positions.at(Session::KeyId(binding.block.key)),
                layer_at * group->group.component_count + component_at, layer_at, component_at};
    for (const auto& prior : slots)
      if (prior.block == key.block && prior.bit == key.bit)
        return writing ? ErrorStatus(KVC_CONFLICT, "overlapping write of one slot")
                       : ErrorStatus(KVC_INVALID_ARGUMENT, "duplicate binding of one slot");
    slots.push_back(key);
    sources.push_back(&binding);
    slot_regions.push_back(region);
  }

  // Everything validated; apply side effects. Store coverage first so a
  // cross-request overlap still fails without touching payloads.
  if (coverage) {
    for (size_t i = 0; i < slots.size(); ++i) {
      std::vector<bool>& bits = coverage->at(slots[i].owned);
      if (bits.size() <= slots[i].bit)
        return ErrorStatus(KVC_INTERNAL, "coverage bitmap is corrupt");
      if (bits[slots[i].bit]) return ErrorStatus(KVC_CONFLICT, "overlapping write of one slot");
    }
    for (size_t i = 0; i < slots.size(); ++i) (*coverage)[slots[i].owned][slots[i].bit] = true;
  }
  if (!writing) {
    // Load destinations must not alias within one request.
    for (size_t i = 0; i < slots.size(); ++i) {
      for (size_t j = i + 1; j < slots.size(); ++j) {
        if (slot_regions[i] != slot_regions[j]) continue;
        const uint64_t a0 = sources[i]->byte_offset;
        const uint64_t a1 = a0 + sources[i]->byte_count;
        const uint64_t b0 = sources[j]->byte_offset;
        const uint64_t b1 = b0 + sources[j]->byte_count;
        if (a0 < b1 && b0 < a1)
          return ErrorStatus(KVC_INVALID_ARGUMENT, "load destinations alias");
      }
    }
  }

  for (size_t i = 0; i < slots.size(); ++i) {
    unsigned char* address =
        reinterpret_cast<unsigned char*>(slot_regions[i]->info.local_address) +
        sources[i]->byte_offset;
    out->entries.push_back(
        {slots[i].block,
         slots[i].layer_at * RowBytes(*group) + group->slot_offsets[slots[i].component_at],
         address, sources[i]->byte_count});
  }
  return OkStatus();
}

namespace {

// Shared acceptance path: validate, retain dependencies and regions, and
// create a transfer. The caller holds the session mutex; the copy step runs
// after acceptance only.
KvcStatus AcceptTransfer(Session* session, std::vector<BlockPtr> blocks,
                         const KvcTransferRequest* request, uint32_t required_access,
                         bool writing, std::vector<std::vector<bool>>* coverage,
                         BindingPlan* plan, Transfer** out) {
  if (!request || request->struct_size < sizeof(KvcTransferRequest))
    return ErrorStatus(KVC_INVALID_ARGUMENT, "transfer request is malformed");
  // Dependencies must be terminal transfers of this session; a failed
  // dependency fails the submission without touching payloads.
  for (uint64_t i = 0; i < request->dependency_count; ++i) {
    KvcObject* dep = request->dependencies[i];
    if (!Session::Owns(dep, KVC_TRANSFER) || Session::AsHeader(dep)->session != session)
      return ErrorStatus(KVC_STALE_HANDLE, "dependency is not a transfer of this session");
    auto* transfer = reinterpret_cast<Transfer*>(dep);
    std::lock_guard<std::mutex> lock(transfer->mutex);
    if (transfer->state != KVC_SUCCEEDED)
      return ErrorStatus(
          transfer->final_code != KVC_OK ? transfer->final_code : (uint32_t)KVC_ABORTED,
          "dependency did not complete successfully");
  }

  if (session->closing) return ErrorStatus(KVC_SHUTTING_DOWN, "session is closing");
  if (KvcStatus status = ResolveBindings(*session, request->selection, request->bindings,
                                         request->binding_count, blocks, required_access,
                                         writing, coverage, plan);
      status.code != KVC_OK)
    return status;

  auto* transfer = new (std::nothrow) Transfer();
  if (!transfer) return ErrorStatus(KVC_RESOURCE_EXHAUSTED, "transfer allocation failed");
  transfer->header.kind = KVC_TRANSFER;
  transfer->header.session = session;
  transfer->blocks = plan->blocks;  // Plan stays usable for the copy step.
  for (uint64_t i = 0; i < request->binding_count; ++i) {
    Header* region = Session::AsHeader(request->bindings[i].region);
    region->refs.fetch_add(1, std::memory_order_relaxed);
    transfer->regions.push_back(region);
  }
  for (uint64_t i = 0; i < request->dependency_count; ++i) {
    Header* dep = Session::AsHeader(request->dependencies[i]);
    dep->refs.fetch_add(1, std::memory_order_relaxed);
    transfer->deps.push_back(dep);
  }
  session->ObjectCreated();
  *out = transfer;
  return OkStatus();
}

}  // namespace

KvcStatus Load(KvcObject* object, KvcObject* read_handle, const KvcTransferRequest* request,
               KvcObject** out) {
  return Guard([&]() -> KvcStatus {
    auto* session = reinterpret_cast<Session*>(object);
    *out = nullptr;
    if (!Session::Owns(read_handle, KVC_READ) ||
        Session::AsHeader(read_handle)->session != session)
      return ErrorStatus(KVC_STALE_HANDLE, "read handle is invalid for this session");
    auto* read = reinterpret_cast<Read*>(read_handle);

    BindingPlan plan;
    Transfer* transfer = nullptr;
    {
      std::lock_guard<std::mutex> lock(session->mutex);
      if (KvcStatus status = AcceptTransfer(session, read->blocks, request, KVC_READ_WRITE,
                                            false, nullptr, &plan, &transfer);
          status.code != KVC_OK)
        return status;
      const uint64_t tick = session->lru_tick.fetch_add(1);
      for (auto& block : plan.blocks) block->last_use = tick;
    }
    // Synchronous copies out of pinned, immutable committed state.
    uint64_t moved = 0;
    for (const auto& entry : plan.entries) {
      std::memcpy(entry.host_address,
                  plan.blocks[entry.block_position]->payload.data() + entry.payload_offset,
                  static_cast<size_t>(entry.byte_count));
      moved += entry.byte_count;
    }
    session->stats_loads.fetch_add(1, std::memory_order_relaxed);
    session->stats_load_bytes.fetch_add(moved, std::memory_order_relaxed);
    *out = reinterpret_cast<KvcObject*>(transfer);
    return OkStatus();
  });
}

KvcStatus Store(KvcObject* object, KvcObject* write_handle, const KvcTransferRequest* request,
                KvcObject** out) {
  return Guard([&]() -> KvcStatus {
    auto* session = reinterpret_cast<Session*>(object);
    *out = nullptr;
    if (!Session::Owns(write_handle, KVC_WRITE) ||
        Session::AsHeader(write_handle)->session != session)
      return ErrorStatus(KVC_STALE_HANDLE, "write handle is invalid for this session");
    auto* write = reinterpret_cast<Write*>(write_handle);

    BindingPlan plan;
    Transfer* transfer = nullptr;
    {
      std::lock_guard<std::mutex> lock(session->mutex);
      if (write->tx->state == KVC_WRITE_ABORTED)
        return ErrorStatus(KVC_ABORTED, "transaction was aborted");
      if (write->tx->state == KVC_COMMITTED)
        return ErrorStatus(KVC_ALREADY_COMMITTED, "transaction is already committed");
      if (KvcStatus status = AcceptTransfer(session, write->tx->blocks, request, 0, true,
                                            &write->tx->covered, &plan, &transfer);
          status.code != KVC_OK)
        return status;
      // Copy under the same lock: coverage and payload move together.
      for (const auto& entry : plan.entries)
        std::memcpy(plan.blocks[entry.block_position]->payload.data() + entry.payload_offset,
                    entry.host_address, static_cast<size_t>(entry.byte_count));
      if (write->tx->state == KVC_ALLOCATED) write->tx->state = KVC_WRITING;
      const uint64_t tick = session->lru_tick.fetch_add(1);
      for (auto& block : plan.blocks) block->last_use = tick;
      uint64_t moved = 0;
      for (const auto& entry : plan.entries) moved += entry.byte_count;
      session->stats_stores.fetch_add(1, std::memory_order_relaxed);
      session->stats_store_bytes.fetch_add(moved, std::memory_order_relaxed);
    }
    *out = reinterpret_cast<KvcObject*>(transfer);
    return OkStatus();
  });
}

KvcStatus PollTransfer(KvcObject* object, KvcTransferStatus* out) {
  return Guard([&]() -> KvcStatus {
    if (!out || out->struct_size < sizeof(*out))
      return ErrorStatus(KVC_INVALID_ARGUMENT, "poll record is too small");
    if (!Session::Owns(object, KVC_TRANSFER))
      return ErrorStatus(KVC_STALE_HANDLE, "object is not a transfer");
    auto* transfer = reinterpret_cast<Transfer*>(object);
    std::lock_guard<std::mutex> lock(transfer->mutex);
    out->state = transfer->state;
    out->result = transfer->final_code == KVC_OK
                      ? KvcStatus{KVC_OK, {}, nullptr, nullptr}
                      : ErrorStatus(transfer->final_code, transfer->final_text);
    return OkStatus();
  });
}

KvcStatus WaitTransfer(KvcObject* object, uint64_t timeout) {
  return Guard([&]() -> KvcStatus {
    if (!Session::Owns(object, KVC_TRANSFER))
      return ErrorStatus(KVC_STALE_HANDLE, "object is not a transfer");
    (void)timeout;  // Transfers here are terminal on acceptance.
    auto* transfer = reinterpret_cast<Transfer*>(object);
    std::lock_guard<std::mutex> lock(transfer->mutex);
    if (transfer->state == KVC_SUCCEEDED) return OkStatus();
    if (transfer->state == KVC_FAILED || transfer->state == KVC_TRANSFER_CANCELLED)
      return ErrorStatus(transfer->final_code, transfer->final_text);
    return ErrorStatus(KVC_DEADLINE_EXCEEDED, "transfer is still pending");
  });
}

}  // namespace kvc_host
