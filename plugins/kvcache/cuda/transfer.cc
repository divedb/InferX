// Data plane of the CUDA provider: memory import, load with promotion of
// offloaded blocks, lazy slot allocation on store, and event-driven
// asynchronous completion.

#include <chrono>
#include <cstring>
#include <new>
#include <thread>
#include <utility>

#include "internal.h"

namespace kvc_cuda {
namespace {

bool StringIs(const KvcString& text, const char* literal) {
  const size_t length = std::strlen(literal);
  return text.size == length && (length == 0 || std::memcmp(text.data, literal, length) == 0);
}

uint64_t RowBytes(const Session::GroupCopy& group) {
  return group.payload_bytes / group.group.layer_count;
}

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

// A transfer plan entry: where the slot bytes live inside the block and
// where they must move, plus the destination's placement.
struct CudaEntry {
  size_t block_position;  // Index into plan->blocks.
  uint64_t payload_offset;
  unsigned char* address;  // Binding address (host or device pointer).
  bool address_is_device;
  uint64_t byte_count;
};

struct CudaPlan {
  const Session::GroupCopy* group = nullptr;
  std::vector<BlockPtr> blocks;
  std::vector<CudaEntry> entries;
  // Slot keys for coverage application (store only), parallel to entries.
  struct SlotKey {
    size_t block_position;
    size_t owned;
    size_t bit;
  };
  std::vector<SlotKey> slots;
};

// Validation shared by load and store; mirrors the host provider's rules.
// Coverage and alias side effects are left to the caller.
KvcStatus ResolveBindings(Session& session, const KvcSpan& span, const KvcBinding* bindings,
                          uint64_t binding_count, const std::vector<BlockPtr>& owned,
                          bool writing, CudaPlan* out) {
  const Session::GroupCopy* group = session.FindGroup(span.group);
  if (!group) return ErrorStatus(KVC_INVALID_ARGUMENT, "span references an unknown group");
  if (!span.layers || span.layer_count == 0)
    return ErrorStatus(KVC_INVALID_ARGUMENT, "span selects no layers");
  if (!span.blocks || span.block_count == 0)
    return ErrorStatus(KVC_INVALID_ARGUMENT, "span selects no blocks");
  if (binding_count && !bindings)
    return ErrorStatus(KVC_INVALID_ARGUMENT, "bindings are missing");

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

  std::unordered_map<uint64_t, size_t> positions;  // Key -> position in owned.
  for (size_t i = 0; i < owned.size(); ++i)
    positions.emplace(Session::KeyId(owned[i]->record.key), i);
  std::unordered_map<uint64_t, size_t> selected;  // Key -> position in plan.
  out->group = group;
  out->blocks.clear();
  out->entries.clear();
  out->slots.clear();
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

  const uint64_t expected = span.block_count * layers.size() * group->group.component_count;
  if (binding_count != expected)
    return ErrorStatus(KVC_INVALID_ARGUMENT, "bindings do not exactly cover the selection");

  std::vector<const KvcBinding*> sources;
  std::vector<Region*> slot_regions;
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
    if (writing) {
      // Sources may be read-only; destinations must be read-write.
    } else if (region->info.access != KVC_READ_WRITE) {
      return ErrorStatus(KVC_INVALID_ARGUMENT, "region access mode is insufficient");
    }
    if (binding.byte_offset > region->info.byte_size ||
        binding.byte_count > region->info.byte_size - binding.byte_offset)
      return ErrorStatus(KVC_INVALID_RANGE, "binding exceeds its region");

    CudaPlan::SlotKey key{slot->second, positions.at(Session::KeyId(binding.block.key)),
                          layer_at * group->group.component_count + component_at};
    for (const auto& prior : out->slots)
      if (prior.block_position == key.block_position && prior.bit == key.bit)
        return writing ? ErrorStatus(KVC_CONFLICT, "overlapping write of one slot")
                       : ErrorStatus(KVC_INVALID_ARGUMENT, "duplicate binding of one slot");
    out->slots.push_back(key);
    sources.push_back(&binding);
    slot_regions.push_back(region);

    const bool device = region->info.memory_type == KVC_DEVICE;
    unsigned char* address =
        reinterpret_cast<unsigned char*>(region->info.local_address) + binding.byte_offset;
    out->entries.push_back({slot->second, offset, address, device, binding.byte_count});
  }
  (void)sources;
  (void)slot_regions;
  return OkStatus();
}

// Attempts to complete a pending transfer; returns its current state.
// Caller holds no locks. Returns without waiting.
uint32_t QueryAndComplete(Transfer* transfer, bool lock_session) {
  uint32_t state = transfer->state.load(std::memory_order_acquire);
  if (state != KVC_PENDING) return state;
  cudaError_t err = transfer->event ? cudaEventQuery(transfer->event) : cudaSuccess;
  if (err == cudaErrorNotReady) return KVC_PENDING;
  uint32_t target = KVC_SUCCEEDED;
  if (err != cudaSuccess) {
    std::lock_guard<std::mutex> lock(transfer->mutex);
    transfer->final_code = KVC_TRANSPORT_ERROR;
    transfer->final_text = "device transfer reported an error";
    target = KVC_FAILED;
  }
  uint32_t expected = KVC_PENDING;
  if (!transfer->state.compare_exchange_strong(expected, target, std::memory_order_release,
                                               std::memory_order_acquire))
    return expected;  // Another observer completed it first.
  if (lock_session) {
    Session* session = transfer->header.session;
    std::lock_guard<std::mutex> lock(session->mutex);
    session->FinishTransferLocked(transfer);
  }
  return target;
}

// Destroys a transfer that was never returned to the caller: unwinds the
// references it took, frees its event, and deletes the shell. Only valid on
// submission-failure paths where no completion bookkeeping applies.
void DestroyTransferShell(Transfer* transfer) {
  std::vector<Header*> regions;
  std::vector<Header*> deps;
  {
    std::lock_guard<std::mutex> lock(transfer->mutex);
    regions = std::move(transfer->regions);
    deps = std::move(transfer->deps);
    transfer->blocks.clear();
    transfer->tx.reset();
  }
  if (transfer->event) cudaEventDestroy(transfer->event);
  delete transfer;
  for (Header* region : regions) ReleaseObject(reinterpret_cast<KvcObject*>(region));
  for (Header* dep : deps) ReleaseObject(reinterpret_cast<KvcObject*>(dep));
}

}  // namespace

uint32_t ObserveTransfer(Transfer* transfer) { return QueryAndComplete(transfer, true); }

uint32_t ObserveTransferLocked(Transfer* transfer) {
  // The caller holds the session mutex; completion bookkeeping runs inline.
  uint32_t state = transfer->state.load(std::memory_order_acquire);
  if (state != KVC_PENDING) return state;
  cudaError_t err = transfer->event ? cudaEventQuery(transfer->event) : cudaSuccess;
  if (err == cudaErrorNotReady) return KVC_PENDING;
  uint32_t target = KVC_SUCCEEDED;
  if (err != cudaSuccess) {
    std::lock_guard<std::mutex> lock(transfer->mutex);
    transfer->final_code = KVC_TRANSPORT_ERROR;
    transfer->final_text = "device transfer reported an error";
    target = KVC_FAILED;
  }
  uint32_t expected = KVC_PENDING;
  if (!transfer->state.compare_exchange_strong(expected, target, std::memory_order_release,
                                               std::memory_order_acquire))
    return expected;
  transfer->header.session->FinishTransferLocked(transfer);
  return target;
}

void DrainTransfer(Transfer* transfer) {
  if (transfer->state.load(std::memory_order_acquire) == KVC_PENDING && transfer->event) {
    cudaSetDevice(transfer->header.session->device_ordinal);
    cudaEventSynchronize(transfer->event);  // Work is already enqueued; bounded.
  }
  ObserveTransfer(transfer);
}

// ---------------------------------------------------------------------------
// Import
// ---------------------------------------------------------------------------

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
    if (info.access != KVC_READ_ONLY && info.access != KVC_READ_WRITE)
      return ErrorStatus(KVC_INVALID_ARGUMENT, "access mode is invalid");
    bool host_memory = info.memory_type == KVC_HOST || info.memory_type == KVC_PINNED_HOST;
    if (host_memory) {
      if (!StringIs(info.device_runtime, "cpu") || !StringIs(info.device_identifier, "0"))
        return ErrorStatus(KVC_UNSUPPORTED, "host imports require local CPU memory");
    } else if (info.memory_type == KVC_DEVICE) {
      const std::string expected = std::to_string(session->device_ordinal);
      if (!StringIs(info.device_runtime, "cuda") ||
          info.device_identifier.size != expected.size() ||
          std::memcmp(info.device_identifier.data, expected.data(), expected.size()) != 0)
        return ErrorStatus(KVC_UNSUPPORTED, "device imports must target this session's GPU");
    } else {
      return ErrorStatus(KVC_UNSUPPORTED, "unsupported memory type");
    }
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

// ---------------------------------------------------------------------------
// Load and store
// ---------------------------------------------------------------------------

namespace {

KvcStatus ValidateDependencies(Session* session, const KvcTransferRequest* request,
                               bool* any_pending) {
  *any_pending = false;
  for (uint64_t i = 0; i < request->dependency_count; ++i) {
    KvcObject* dep = request->dependencies[i];
    if (!Session::Owns(dep, KVC_TRANSFER) || Session::AsHeader(dep)->session != session)
      return ErrorStatus(KVC_STALE_HANDLE, "dependency is not a transfer of this session");
    auto* transfer = reinterpret_cast<Transfer*>(dep);
    const uint32_t state = ObserveTransferLocked(transfer);
    if (state == KVC_PENDING) {
      *any_pending = true;
      continue;
    }
    if (state != KVC_SUCCEEDED) {
      std::lock_guard<std::mutex> lock(transfer->mutex);
      const uint32_t code =
          transfer->final_code != KVC_OK ? transfer->final_code : (uint32_t)KVC_ABORTED;
      std::string text = transfer->final_text;
      return ErrorStatus(code, "dependency failed: " + text);
    }
  }
  return OkStatus();
}

}  // namespace

KvcStatus Load(KvcObject* object, KvcObject* read_handle, const KvcTransferRequest* request,
               KvcObject** out) {
  return Guard([&]() -> KvcStatus {
    auto* session = reinterpret_cast<Session*>(object);
    *out = nullptr;
    if (!request || request->struct_size < sizeof(KvcTransferRequest))
      return ErrorStatus(KVC_INVALID_ARGUMENT, "transfer request is malformed");
    if (!Session::Owns(read_handle, KVC_READ) ||
        Session::AsHeader(read_handle)->session != session)
      return ErrorStatus(KVC_STALE_HANDLE, "read handle is invalid for this session");
    auto* read = reinterpret_cast<Read*>(read_handle);

    std::lock_guard<std::mutex> lock(session->mutex);
    if (session->closing) return ErrorStatus(KVC_SHUTTING_DOWN, "session is closing");
    bool any_pending = false;
    if (KvcStatus status = ValidateDependencies(session, request, &any_pending);
        status.code != KVC_OK)
      return status;

    CudaPlan plan;
    if (KvcStatus status = ResolveBindings(*session, request->selection, request->bindings,
                                           request->binding_count, read->blocks, false, &plan);
        status.code != KVC_OK)
      return status;

    cudaSetDevice(session->device_ordinal);
    // Any device destination justifies promoting offloaded blocks back into
    // GPU slots; a failed promotion falls back to host-to-device copies.
    bool any_device_destination = false;
    for (const auto& entry : plan.entries) any_device_destination |= entry.address_is_device;
    if (any_device_destination) {
      for (auto& block : plan.blocks)
        if (!block->gpu) session->TryPromoteLocked(block);
    }
    for (auto& block : plan.blocks)
      if (!block->gpu && block->host.empty())
        return ErrorStatus(KVC_INTERNAL, "committed block has no storage");
    // Host-addressable copies run synchronously after any pending same-
    // stream dependencies; pure device copies are stream-ordered anyway.
    bool needs_host_sync = any_pending;
    for (const auto& entry : plan.entries) {
      if (!entry.address_is_device) needs_host_sync = true;
      if (!plan.blocks[entry.block_position]->gpu) needs_host_sync = true;
    }
    if (needs_host_sync && cudaStreamSynchronize(session->stream) != cudaSuccess)
      return ErrorStatus(KVC_TRANSPORT_ERROR, "stream synchronization failed");

    auto* transfer = new (std::nothrow) Transfer();
    if (!transfer) return ErrorStatus(KVC_RESOURCE_EXHAUSTED, "transfer allocation failed");
    cudaEvent_t event = nullptr;
    if (cudaEventCreateWithFlags(&event, cudaEventDisableTiming) != cudaSuccess) {
      delete transfer;
      return ErrorStatus(KVC_RESOURCE_EXHAUSTED, "event creation failed");
    }
    transfer->header.kind = KVC_TRANSFER;
    transfer->header.session = session;
    transfer->event = event;
    transfer->blocks = plan.blocks;
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

    bool enqueued_async = false;
    for (const auto& entry : plan.entries) {
      const BlockPtr& block = plan.blocks[entry.block_position];
      const void* source;
      if (block->gpu) {
        source = static_cast<const unsigned char*>(block->gpu) + entry.payload_offset;
      } else {
        source = block->host.data() + entry.payload_offset;
      }
      cudaError_t err;
      if (entry.address_is_device) {
        err = cudaMemcpyAsync(entry.address, source, entry.byte_count,
                              block->gpu ? cudaMemcpyDeviceToDevice : cudaMemcpyHostToDevice,
                              session->stream);
        enqueued_async = true;
      } else if (block->gpu) {
        err = cudaMemcpyAsync(entry.address, source, entry.byte_count, cudaMemcpyDeviceToHost,
                              session->stream);
        enqueued_async = true;
      } else {
        std::memcpy(entry.address, source, static_cast<size_t>(entry.byte_count));
        err = cudaSuccess;
      }
      if (err != cudaSuccess) {
        cudaStreamSynchronize(session->stream);
        DestroyTransferShell(transfer);  // Frees the event and the shell.
        return ErrorStatus(KVC_TRANSPORT_ERROR, "load enqueue failed");
      }
    }
    if (enqueued_async && cudaEventRecord(event, session->stream) != cudaSuccess) {
      cudaStreamSynchronize(session->stream);
      DestroyTransferShell(transfer);
      return ErrorStatus(KVC_TRANSPORT_ERROR, "event record failed");
    }

    for (auto& block : plan.blocks) ++block->pending;
    const uint64_t tick = session->lru_tick.fetch_add(1);
    for (auto& block : plan.blocks) block->last_use = tick;
    uint64_t moved = 0;
    for (const auto& entry : plan.entries) moved += entry.byte_count;
    session->stats_loads.fetch_add(1, std::memory_order_relaxed);
    session->stats_load_bytes.fetch_add(moved, std::memory_order_relaxed);
    if (enqueued_async) {
      cudaEventRecord(event, session->stream);
    } else {
      // Everything completed synchronously: finish the accounting inline.
      transfer->state.store(KVC_SUCCEEDED, std::memory_order_release);
      session->FinishTransferLocked(transfer);
    }
    session->ObjectCreated();
    *out = reinterpret_cast<KvcObject*>(transfer);
    return OkStatus();
  });
}

KvcStatus Store(KvcObject* object, KvcObject* write_handle, const KvcTransferRequest* request,
                KvcObject** out) {
  return Guard([&]() -> KvcStatus {
    auto* session = reinterpret_cast<Session*>(object);
    *out = nullptr;
    if (!request || request->struct_size < sizeof(KvcTransferRequest))
      return ErrorStatus(KVC_INVALID_ARGUMENT, "transfer request is malformed");
    if (!Session::Owns(write_handle, KVC_WRITE) ||
        Session::AsHeader(write_handle)->session != session)
      return ErrorStatus(KVC_STALE_HANDLE, "write handle is invalid for this session");
    auto* write = reinterpret_cast<Write*>(write_handle);

    std::lock_guard<std::mutex> lock(session->mutex);
    if (session->closing) return ErrorStatus(KVC_SHUTTING_DOWN, "session is closing");
    if (write->tx->state == KVC_WRITE_ABORTED)
      return ErrorStatus(KVC_ABORTED, "transaction was aborted");
    if (write->tx->state == KVC_COMMITTED)
      return ErrorStatus(KVC_ALREADY_COMMITTED, "transaction is already committed");

    bool any_pending = false;
    if (KvcStatus status = ValidateDependencies(session, request, &any_pending);
        status.code != KVC_OK)
      return status;
    // Host sources stage synchronously into the stream, so pending earlier
    // work must complete before their bytes are read.
    if (any_pending && cudaStreamSynchronize(session->stream) != cudaSuccess)
      return ErrorStatus(KVC_TRANSPORT_ERROR, "stream synchronization failed");

    CudaPlan plan;
    if (KvcStatus status =
            ResolveBindings(*session, request->selection, request->bindings,
                            request->binding_count, write->tx->blocks, true, &plan);
        status.code != KVC_OK)
      return status;
    // Reject cross-request overlap before allocating anything.
    for (const auto& slot : plan.slots) {
      auto& bits = write->tx->covered.at(slot.owned);
      if (bits.size() <= slot.bit)
        return ErrorStatus(KVC_INTERNAL, "coverage bitmap is corrupt");
      if (bits[slot.bit]) return ErrorStatus(KVC_CONFLICT, "overlapping write of one slot");
    }

    cudaSetDevice(session->device_ordinal);
    // Allocate device slots for blocks that do not have one yet. Begin-write
    // already reserved and accounted for these bytes, so allocation converts
    // reservation into a slot rather than claiming new budget; slots are
    // returned if a later step fails so the transaction stays unchanged.
    std::vector<BlockPtr> freshly_allocated;
    for (auto& block : plan.blocks) {
      if (block->gpu) continue;
      SlotPool* pool = session->PoolForLocked(block->payload_bytes);
      void* slot = pool->Take();
      if (!slot) {
        for (auto& allocated : freshly_allocated) {
          session->ReclaimLocked(allocated);
          write->tx->reserved_remainder += allocated->payload_bytes;
          session->gpu_reserved += allocated->payload_bytes;
        }
        return ErrorStatus(KVC_RESOURCE_EXHAUSTED, "device slot allocation failed");
      }
      block->gpu = slot;
      block->pool = session->pools.at(block->payload_bytes);
      block->session = session;
      // The reservation becomes a live slot; pool accounting replaces it.
      write->tx->reserved_remainder -= block->payload_bytes;
      session->gpu_reserved -= block->payload_bytes;
      freshly_allocated.push_back(block);
    }

    auto* transfer = new (std::nothrow) Transfer();
    if (!transfer) {
      for (auto& allocated : freshly_allocated) {
        session->ReclaimLocked(allocated);
        write->tx->reserved_remainder += allocated->payload_bytes;
        session->gpu_reserved += allocated->payload_bytes;
      }
      return ErrorStatus(KVC_RESOURCE_EXHAUSTED, "transfer allocation failed");
    }
    cudaEvent_t event = nullptr;
    if (cudaEventCreateWithFlags(&event, cudaEventDisableTiming) != cudaSuccess) {
      delete transfer;
      for (auto& allocated : freshly_allocated) {
        session->ReclaimLocked(allocated);
        write->tx->reserved_remainder += allocated->payload_bytes;
        session->gpu_reserved += allocated->payload_bytes;
      }
      return ErrorStatus(KVC_RESOURCE_EXHAUSTED, "event creation failed");
    }
    transfer->header.kind = KVC_TRANSFER;
    transfer->header.session = session;
    transfer->event = event;
    transfer->poisoned_tx = true;  // A later failure aborts the transaction.
    transfer->tx = write->tx;
    transfer->blocks = plan.blocks;
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

    for (const auto& entry : plan.entries) {
      const BlockPtr& block = plan.blocks[entry.block_position];
      void* destination = static_cast<unsigned char*>(block->gpu) + entry.payload_offset;
      const cudaError_t err = cudaMemcpyAsync(
          destination, entry.address, entry.byte_count,
          entry.address_is_device ? cudaMemcpyDeviceToDevice : cudaMemcpyHostToDevice,
          session->stream);
      if (err != cudaSuccess) {
        cudaStreamSynchronize(session->stream);
        DestroyTransferShell(transfer);
        for (auto& allocated : freshly_allocated) {
          session->ReclaimLocked(allocated);
          write->tx->reserved_remainder += allocated->payload_bytes;
          session->gpu_reserved += allocated->payload_bytes;
        }
        return ErrorStatus(KVC_TRANSPORT_ERROR, "store enqueue failed");
      }
    }

    // Accepted: apply coverage, retention, and transaction state.
    uint64_t moved = 0;
    for (const auto& entry : plan.entries) moved += entry.byte_count;
    session->stats_stores.fetch_add(1, std::memory_order_relaxed);
    session->stats_store_bytes.fetch_add(moved, std::memory_order_relaxed);
    for (const auto& slot : plan.slots) write->tx->covered[slot.owned][slot.bit] = true;
    for (auto& block : plan.blocks) ++block->pending;
    const uint64_t tick = session->lru_tick.fetch_add(1);
    for (auto& block : plan.blocks) block->last_use = tick;
    if (write->tx->state == KVC_ALLOCATED) write->tx->state = KVC_WRITING;
    write->tx->transfers.push_back(&transfer->header);
    if (cudaEventRecord(event, session->stream) != cudaSuccess) {
      cudaStreamSynchronize(session->stream);
      write->tx->transfers.pop_back();
      for (auto& block : plan.blocks) --block->pending;
      for (const auto& slot : plan.slots) write->tx->covered[slot.owned][slot.bit] = false;
      DestroyTransferShell(transfer);
      for (auto& allocated : freshly_allocated) {
        session->ReclaimLocked(allocated);
        write->tx->reserved_remainder += allocated->payload_bytes;
        session->gpu_reserved += allocated->payload_bytes;
      }
      return ErrorStatus(KVC_TRANSPORT_ERROR, "event record failed");
    }
    session->ObjectCreated();
    *out = reinterpret_cast<KvcObject*>(transfer);
    return OkStatus();
  });
}

// ---------------------------------------------------------------------------
// Completion table
// ---------------------------------------------------------------------------

KvcStatus PollTransfer(KvcObject* object, KvcTransferStatus* out) {
  return Guard([&]() -> KvcStatus {
    if (!out || out->struct_size < sizeof(*out))
      return ErrorStatus(KVC_INVALID_ARGUMENT, "poll record is too small");
    if (!Session::Owns(object, KVC_TRANSFER))
      return ErrorStatus(KVC_STALE_HANDLE, "object is not a transfer");
    auto* transfer = reinterpret_cast<Transfer*>(object);
    const uint32_t state = ObserveTransfer(transfer);
    std::lock_guard<std::mutex> lock(transfer->mutex);
    out->state = state;
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
    auto* transfer = reinterpret_cast<Transfer*>(object);
    const bool bounded = timeout != KVC_TIMEOUT_INFINITE;
    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::nanoseconds(bounded ? timeout : 0);
    for (;;) {
      const uint32_t state = ObserveTransfer(transfer);
      if (state == KVC_SUCCEEDED) return OkStatus();
      if (state == KVC_FAILED || state == KVC_TRANSFER_CANCELLED) {
        std::lock_guard<std::mutex> lock(transfer->mutex);
        return transfer->final_code == KVC_OK
                   ? ErrorStatus(KVC_INTERNAL, "terminal transfer lacks a failure code")
                   : ErrorStatus(transfer->final_code, transfer->final_text);
      }
      if (bounded && std::chrono::steady_clock::now() >= deadline)
        return ErrorStatus(KVC_DEADLINE_EXCEEDED, "transfer is still pending");
      std::this_thread::sleep_for(std::chrono::microseconds(100));
    }
  });
}

}  // namespace kvc_cuda
