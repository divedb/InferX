// Test-only provider with genuinely asynchronous transfers: accepted work
// executes on a background thread after a configurable delay, dependencies
// are awaited before payload access, and a chosen store fails in flight to
// exercise poisoned transactions and failed dependencies.
//
// Options schema "async.fixture", 8 canonical bytes:
//   u32 delay_ms (default 25)
//   u32 fail_store (1-based index among accepted stores; 0 disables)

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <new>
#include <queue>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "kvc/abi/provider.h"

namespace {

constexpr uint32_t kMagic = 0x4b564341u;  // "KVCA"
constexpr uint64_t kBudgetBytes = 64ull << 20;

struct Session;
struct Block;
struct TxRecord;
struct Transfer;

using BlockPtr = std::shared_ptr<Block>;
using TxPtr = std::shared_ptr<TxRecord>;

struct Header {
  std::atomic<uint64_t> refs{1};
  uint32_t kind = 0;
  uint32_t magic = kMagic;
  Session* session = nullptr;
};

KvcStatus OkStatus() { return KvcStatus{KVC_OK, {}, nullptr, nullptr}; }

KvcStatus ErrorStatus(uint32_t code, std::string diagnostic) {
  auto* text = new (std::nothrow) std::string(std::move(diagnostic));
  if (!text) return KvcStatus{code, {}, nullptr, nullptr};
  return KvcStatus{
      code, {reinterpret_cast<const char*>(text->data()), text->size()}, text, [](void* owner) {
        delete static_cast<std::string*>(owner);
      }};
}

struct Block {
  KvcBlock record{};
  std::vector<unsigned char> payload;
};

struct Region {
  Header header{};
  std::string runtime;
  std::string identifier;
  KvcMemoryInfo info{};
  void* owner_context = nullptr;
  void (*owner_release)(void*) = nullptr;
};

struct Read {
  Header header{};
  std::vector<KvcBlock> manifest;
  std::vector<BlockPtr> blocks;
};

struct Write {
  Header header{};
  TxPtr tx;
};

struct TxRecord {
  KvcTransactionId id{};
  uint32_t state = KVC_ALLOCATED;
  std::vector<KvcBlock> manifest;
  std::vector<BlockPtr> blocks;
  std::vector<std::vector<bool>> covered;
  std::vector<Header*> transfers;  // Pending store transfers.
  uint64_t bytes = 0;
};

// One enqueued copy: fixed host addresses captured at submission.
struct Copy {
  unsigned char* dst;
  const unsigned char* src;
  uint64_t bytes;
};

struct WorkItem;

struct Transfer {
  Header header{};
  std::atomic<uint32_t> state{KVC_PENDING};
  uint32_t final_code = KVC_OK;
  std::string final_text;
  bool fail = false;  // Injected failure; no copies run.
  bool store = false;
  TxPtr tx;  // Store transfers poison on failure.
  std::vector<BlockPtr> blocks;
  std::vector<Header*> regions;
  std::vector<Header*> deps;
  std::vector<Copy> copies;
};

struct Session {
  Header header{};
  std::mutex mutex;
  std::condition_variable work_signal;
  bool closing = false;
  bool configured = false;
  uint32_t delay_ms = 25;
  uint32_t fail_store = 0;
  uint64_t accepted_stores = 0;

  KvcConfig config{};
  std::string partition_name;
  std::vector<unsigned char> partition_payload;
  struct GroupCopy {
    KvcGroup group{};
    std::vector<uint32_t> layers;
    std::vector<KvcComponent> components;
    std::vector<std::string> component_names;
    std::vector<uint64_t> slot_offsets;
    uint64_t payload_bytes = 0;
  };
  std::vector<GroupCopy> groups;
  std::vector<KvcGroup> group_values;  // Contiguous values the borrowed config points into.

  std::unordered_map<uint64_t, BlockPtr> index;
  std::unordered_map<uint64_t, TxPtr> transactions;
  std::unordered_set<uint64_t> reserved_keys;
  uint64_t bytes_used = 0;
  std::atomic<uint64_t> objects_{0};

  // Pending work queue and the worker thread.
  std::deque<WorkItem*> queue;
  std::thread worker;
  bool worker_started = false;
  size_t in_flight = 0;

  ~Session();
};

struct WorkItem {
  Transfer* transfer;
  explicit WorkItem(Transfer* t) : transfer(t) {}
};

template <class Fn>
KvcStatus Guard(Fn&& fn) noexcept {
  try {
    return fn();
  } catch (const std::bad_alloc&) {
    return ErrorStatus(KVC_RESOURCE_EXHAUSTED, "async fixture allocation failed");
  } catch (...) {
    return ErrorStatus(KVC_INTERNAL, "async fixture threw an exception");
  }
}

uint64_t KeyId(const KvcDigest& key) {
  uint64_t view = 0;
  std::memcpy(&view, key.bytes, sizeof(view));
  return view;
}
uint64_t TxId(const KvcTransactionId& id) {
  uint64_t view = 0;
  std::memcpy(&view, id.bytes, sizeof(view));
  return view;
}

Header* AsHeader(KvcObject* object) { return reinterpret_cast<Header*>(object); }
bool Owns(const KvcObject* object, uint32_t kind) {
  if (!object) return false;
  const auto* header = reinterpret_cast<const Header*>(object);
  return header->magic == kMagic && header->kind == kind;
}

// ---------------------------------------------------------------------------
// Worker thread
// ---------------------------------------------------------------------------

void Retain(KvcObject* object);
void Release(KvcObject* object);
void FinishTransfer(Session* session, Transfer* transfer);

void WorkerLoop(Session* session) {
  std::unique_lock<std::mutex> lock(session->mutex);
  for (;;) {
    while (session->queue.empty() && !session->closing) session->work_signal.wait(lock);
    if (session->queue.empty() && session->closing) break;
    WorkItem* item = session->queue.front();
    session->queue.pop_front();
    ++session->in_flight;

    // Wait for dependencies to complete, without holding the lock.
    std::vector<Header*> deps = item->transfer->deps;
    lock.unlock();
    bool dep_failed = false;
    for (Header* dep : deps) {
      auto* dependency = reinterpret_cast<Transfer*>(dep);
      // Dependencies are scheduled before their dependents, so this spin is
      // bounded by the dependency's own delay.
      while (dependency->state.load(std::memory_order_acquire) == KVC_PENDING)
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
      if (dependency->state.load(std::memory_order_acquire) != KVC_SUCCEEDED) {
        dep_failed = true;
        break;
      }
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(session->delay_ms));

    lock.lock();
    if (dep_failed || item->transfer->fail) {
      item->transfer->final_code = dep_failed ? KVC_ABORTED : KVC_TRANSPORT_ERROR;
      item->transfer->final_text =
          dep_failed ? "dependency did not complete successfully" : "injected store failure";
    } else {
      for (const auto& copy : item->transfer->copies)
        std::memcpy(copy.dst, copy.src, static_cast<size_t>(copy.bytes));
      item->transfer->final_code = KVC_OK;
    }
    item->transfer->state.store(dep_failed || item->transfer->fail ? KVC_FAILED : KVC_SUCCEEDED,
                                std::memory_order_release);
    FinishTransfer(session, item->transfer);
    --session->in_flight;
    session->work_signal.notify_all();
    // Release the queue's reference without holding the session lock: the
    // release path itself takes that lock for retained regions.
    Transfer* completed = item->transfer;
    delete item;
    lock.unlock();
    Release(reinterpret_cast<KvcObject*>(completed));
    lock.lock();
  }
}

// Completion bookkeeping; the session lock is held.
void FinishTransfer(Session* session, Transfer* transfer) {
  if (transfer->tx) {
    auto& pending = transfer->tx->transfers;
    for (size_t i = 0; i < pending.size(); ++i) {
      if (pending[i] == &transfer->header) {
        pending[i] = pending.back();
        pending.pop_back();
        break;
      }
    }
    if (transfer->final_code != KVC_OK && transfer->tx->state != KVC_COMMITTED &&
        transfer->tx->state != KVC_WRITE_ABORTED) {
      // Poison: an accepted store failed in flight.
      for (const auto& block : transfer->tx->blocks) {
        session->reserved_keys.erase(KeyId(block->record.key));
        session->bytes_used -= block->payload.size();
      }
      transfer->tx->state = KVC_WRITE_ABORTED;
      transfer->tx->covered.clear();
      transfer->tx->blocks.clear();
    }
  }
}

Session::~Session() {
  {
    std::lock_guard<std::mutex> lock(mutex);
    closing = true;
    work_signal.notify_all();
  }
  if (worker_started && worker.joinable()) worker.join();
  for (WorkItem* item : queue) {
    Release(reinterpret_cast<KvcObject*>(item->transfer));
    delete item;
  }
}

// ---------------------------------------------------------------------------
// Shared object plumbing
// ---------------------------------------------------------------------------

void Retain(KvcObject* object) {
  if (object) AsHeader(object)->refs.fetch_add(1, std::memory_order_relaxed);
}

void AbortLocked(Session& session, const TxPtr& tx) {
  if (tx->state == KVC_COMMITTED || tx->state == KVC_WRITE_ABORTED) return;
  for (const auto& block : tx->blocks) {
    session.reserved_keys.erase(KeyId(block->record.key));
    session.bytes_used -= block->payload.size();
  }
  tx->state = KVC_WRITE_ABORTED;
  tx->covered.clear();
  tx->blocks.clear();
}

void Release(KvcObject* object) {
  if (!object) return;
  auto* header = AsHeader(object);
  if (header->magic != kMagic) return;
  if (header->refs.fetch_sub(1, std::memory_order_acq_rel) != 1) return;
  Session* session = header->session;
  switch (header->kind) {
    case KVC_SESSION: {
      if (session->header.refs.fetch_sub(1, std::memory_order_acq_rel) == 1) delete session;
      return;
    }
    case KVC_READ:
      delete reinterpret_cast<Read*>(object);
      break;
    case KVC_WRITE: {
      auto* write = reinterpret_cast<Write*>(object);
      if (write->tx) {
        std::lock_guard<std::mutex> lock(session->mutex);
        AbortLocked(*session, write->tx);
        write->tx.reset();
      }
      delete write;
      break;
    }
    case KVC_MEMORY: {
      auto* region = reinterpret_cast<Region*>(object);
      if (region->owner_release) region->owner_release(region->owner_context);
      delete region;
      break;
    }
    case KVC_TRANSFER: {
      auto* transfer = reinterpret_cast<Transfer*>(object);
      // A dropped pending handle does not cancel the work; the queue keeps
      // the transfer alive until it completes. Retained regions and
      // dependencies are released here, after the payload copies finished.
      std::vector<Header*> regions;
      std::vector<Header*> deps;
      {
        std::lock_guard<std::mutex> lock(transfer->header.session->mutex);
        regions = std::move(transfer->regions);
        deps = std::move(transfer->deps);
        transfer->blocks.clear();
        transfer->tx.reset();
      }
      delete transfer;
      for (Header* region : regions) Release(reinterpret_cast<KvcObject*>(region));
      for (Header* dep : deps) Release(reinterpret_cast<KvcObject*>(dep));
      break;
    }
    default:
      return;
  }
  session->objects_.fetch_sub(1, std::memory_order_acq_rel);
}

uint32_t Kind(const KvcObject* object) {
  if (!object) return 0;
  const auto* header = reinterpret_cast<const Header*>(object);
  return header->magic == kMagic ? header->kind : 0;
}

// ---------------------------------------------------------------------------
// Negotiation
// ---------------------------------------------------------------------------

KvcStatus KVC_CALL Open(const KvcDescriptor* options, uint64_t timeout, KvcObject** out) {
  (void)timeout;
  return Guard([&]() -> KvcStatus {
    if (!out) return ErrorStatus(KVC_INVALID_ARGUMENT, "output pointer is null");
    *out = nullptr;
    auto* session = new (std::nothrow) Session();
    if (!session) return ErrorStatus(KVC_RESOURCE_EXHAUSTED, "session allocation failed");
    if (options && options->schema.name.size != 0) {
      if (options->schema.name.size != 13 ||
          std::memcmp(options->schema.name.data, "async.fixture", 13) != 0 ||
          options->schema.major != 1)
        return ErrorStatus(KVC_INVALID_ARGUMENT, "unknown options schema");
      if (options->canonical_payload.size == 8) {
        uint32_t delay = 0, fail = 0;
        std::memcpy(&delay, options->canonical_payload.data, 4);
        std::memcpy(&fail, options->canonical_payload.data + 4, 4);
        session->delay_ms = delay;
        session->fail_store = fail;
      } else if (options->canonical_payload.size != 0) {
        return ErrorStatus(KVC_INVALID_ARGUMENT, "options payload must be 8 bytes");
      }
    }
    session->header.kind = KVC_SESSION;
    session->header.session = session;
    *out = reinterpret_cast<KvcObject*>(session);
    return OkStatus();
  });
}

KvcStatus KVC_CALL Capabilities(KvcObject*, KvcCapabilities* out) {
  return Guard([&]() -> KvcStatus {
    if (!out || out->struct_size < sizeof(*out))
      return ErrorStatus(KVC_INVALID_ARGUMENT, "capabilities record is too small");
    static const KvcSchema schemas[] = {{{"kvc.opaque", 10}, 1, 0}};
    out->features = KVC_FEATURE_LAYERWISE;
    out->memory_types = (1u << KVC_HOST) | (1u << KVC_PINNED_HOST);
    out->layouts = 1u << KVC_OPAQUE_BYTES;
    out->schemas = schemas;
    out->schema_count = 1;
    return OkStatus();
  });
}

bool SchemaIs(const KvcSchema& schema, const char* name, uint32_t major) {
  const size_t length = std::strlen(name);
  return schema.major == major && schema.name.size == length &&
         (length == 0 || std::memcmp(schema.name.data, name, length) == 0);
}

KvcStatus KVC_CALL Configure(KvcObject* object, const KvcConfig* request, uint64_t) {
  return Guard([&]() -> KvcStatus {
    auto* session = reinterpret_cast<Session*>(object);
    if (!request || request->struct_size < sizeof(KvcConfig))
      return ErrorStatus(KVC_INVALID_ARGUMENT, "configuration record is too small");
    std::lock_guard<std::mutex> lock(session->mutex);
    if (session->configured || session->closing)
      return ErrorStatus(KVC_INVALID_ARGUMENT, "session is not configurable");
    if (request->required_features & ~static_cast<uint64_t>(KVC_FEATURE_LAYERWISE))
      return ErrorStatus(KVC_UNSUPPORTED, "a required feature is not supported");
    if (request->required_extension_count)
      return ErrorStatus(KVC_UNSUPPORTED, "no extensions are supported");
    if (!request->groups || request->group_count == 0)
      return ErrorStatus(KVC_INVALID_ARGUMENT, "at least one group is required");

    std::vector<Session::GroupCopy> built;
    std::vector<uint32_t> seen;
    for (uint64_t g = 0; g < request->group_count; ++g) {
      const KvcGroup& group = request->groups[g];
      if (!group.layers || !group.layer_count || !group.components || !group.component_count ||
          group.tokens_per_block == 0)
        return ErrorStatus(KVC_INVALID_ARGUMENT, "group description is incomplete");
      if (!SchemaIs(group.semantics.schema, "kvc.opaque", 1))
        return ErrorStatus(KVC_UNSUPPORTED, "group schema must be kvc.opaque v1");
      for (uint32_t id : seen)
        if (id == group.id)
          return ErrorStatus(KVC_INVALID_ARGUMENT, "group ids must be unique");
      seen.push_back(group.id);
      Session::GroupCopy stored;
      stored.group.id = group.id;
      stored.group.tokens_per_block = group.tokens_per_block;
      stored.layers.assign(group.layers, group.layers + group.layer_count);
      for (size_t i = 0; i < stored.layers.size(); ++i)
        if (i && stored.layers[i - 1] >= stored.layers[i])
          return ErrorStatus(KVC_INVALID_ARGUMENT, "layers must be ascending and unique");
      uint64_t row = 0;
      for (uint64_t c = 0; c < group.component_count; ++c) {
        const KvcComponent& component = group.components[c];
        if (component.layout != KVC_OPAQUE_BYTES ||
            !SchemaIs(component.meaning.schema, "kvc.bytes", 1) ||
            component.meaning.canonical_payload.size != 0 || component.bytes_per_block == 0)
          return ErrorStatus(KVC_UNSUPPORTED, "component is outside the opaque profile");
        for (const auto& prior : stored.components)
          if (prior.id == component.id)
            return ErrorStatus(KVC_INVALID_ARGUMENT, "component ids must be unique");
        stored.slot_offsets.push_back(row);
        row += component.bytes_per_block;
        stored.components.push_back(component);
        stored.component_names.emplace_back(component.meaning.schema.name.data,
                                            component.meaning.schema.name.size);
      }
      stored.payload_bytes = row * group.layer_count;
      built.push_back(std::move(stored));
    }
    session->groups = std::move(built);
    session->group_values.clear();
    for (auto& group : session->groups) {
      group.group.layers = group.layers.data();
      group.group.layer_count = group.layers.size();
      group.group.components = group.components.data();
      group.group.component_count = group.components.size();
      session->group_values.push_back(group.group);
    }
    session->partition_name.assign(request->logical_partition.schema.name.data,
                                   request->logical_partition.schema.name.size);
    session->partition_payload.assign(request->logical_partition.canonical_payload.data,
                                      request->logical_partition.canonical_payload.data +
                                          request->logical_partition.canonical_payload.size);
    session->config = *request;
    session->config.struct_size = sizeof(KvcConfig);
    session->config.groups = session->group_values.data();
    session->config.group_count = session->group_values.size();
    session->config.logical_partition.schema.name = {session->partition_name.data(),
                                                     session->partition_name.size()};
    session->config.logical_partition.canonical_payload = {session->partition_payload.data(),
                                                           session->partition_payload.size()};
    session->configured = true;
    return OkStatus();
  });
}

KvcStatus KVC_CALL Configuration(KvcObject* object, const KvcConfig** out) {
  return Guard([&]() -> KvcStatus {
    auto* session = reinterpret_cast<Session*>(object);
    if (!out) return ErrorStatus(KVC_INVALID_ARGUMENT, "output pointer is null");
    std::lock_guard<std::mutex> lock(session->mutex);
    if (!session->configured) return ErrorStatus(KVC_UNAVAILABLE, "session is not configured");
    *out = &session->config;
    return OkStatus();
  });
}

KvcStatus KVC_CALL Shutdown(KvcObject* object, uint64_t) {
  return Guard([&]() -> KvcStatus {
    auto* session = reinterpret_cast<Session*>(object);
    std::unique_lock<std::mutex> lock(session->mutex);
    session->closing = true;
    session->work_signal.notify_all();
    if (session->objects_.load(std::memory_order_acquire) != 0)
      return ErrorStatus(KVC_BUSY, "session objects are still live");
    // No objects remain, so no accepted work can be pending either.
    if (session->worker_started) {
      lock.unlock();
      if (session->worker.joinable()) session->worker.join();
    }
    return OkStatus();
  });
}

// ---------------------------------------------------------------------------
// Control plane (mirrors the host reference provider)
// ---------------------------------------------------------------------------

const Session::GroupCopy* FindGroup(Session* session, uint32_t id) {
  for (const auto& group : session->groups)
    if (group.group.id == id) return &group;
  return nullptr;
}

const Session::GroupCopy* ValidateBlock(Session* session, const KvcBlock& block,
                                        KvcStatus* error) {
  *error = OkStatus();
  const Session::GroupCopy* group = FindGroup(session, block.group);
  if (!group) {
    *error = ErrorStatus(KVC_INVALID_ARGUMENT, "block references an unknown group");
    return nullptr;
  }
  if (block.first_token != block.logical_index * group->group.tokens_per_block ||
      block.token_count == 0 || block.token_count > group->group.tokens_per_block) {
    *error = ErrorStatus(KVC_INVALID_RANGE, "block token extent is inconsistent");
    return nullptr;
  }
  return group;
}

bool SameRecord(const KvcBlock& a, const KvcBlock& b) {
  return a.group == b.group && a.logical_index == b.logical_index &&
         a.first_token == b.first_token && a.token_count == b.token_count &&
         std::memcmp(a.dependency.bytes, b.dependency.bytes, sizeof(a.dependency.bytes)) == 0;
}

KvcStatus KVC_CALL Lookup(KvcObject* object, const KvcLookupRequest* request, uint64_t,
                          KvcLookupResult* out) {
  return Guard([&]() -> KvcStatus {
    auto* session = reinterpret_cast<Session*>(object);
    if (!request || request->struct_size < sizeof(KvcLookupRequest) || !out)
      return ErrorStatus(KVC_INVALID_ARGUMENT, "lookup request is malformed");
    out->read = nullptr;
    out->prefix = KvcPrefix{};
    std::lock_guard<std::mutex> lock(session->mutex);
    if (session->closing) return ErrorStatus(KVC_SHUTTING_DOWN, "session is closing");
    if (!session->configured) return ErrorStatus(KVC_UNAVAILABLE, "session is not configured");
    std::unordered_set<uint64_t> keys;
    for (uint64_t i = 0; i < request->catalog_count; ++i) {
      KvcStatus error;
      if (!ValidateBlock(session, request->catalog[i], &error)) return error;
      if (!keys.insert(KeyId(request->catalog[i].key)).second)
        return ErrorStatus(KVC_INVALID_ARGUMENT, "catalog keys must be unique");
    }
    const KvcCandidate* best = nullptr;
    std::vector<BlockPtr> best_blocks;
    for (uint64_t c = 0; c < request->candidate_count; ++c) {
      const KvcCandidate& candidate = request->candidates[c];
      std::vector<BlockPtr> resolved;
      bool available = true;
      for (uint64_t r = 0; r < candidate.required_count && available; ++r) {
        const KvcRange& range = candidate.required[r];
        if (range.first > request->catalog_count ||
            range.count > request->catalog_count - range.first)
          return ErrorStatus(KVC_INVALID_RANGE, "candidate range exceeds the catalog");
        for (uint64_t i = 0; i < range.count; ++i) {
          const KvcBlock& want = request->catalog[range.first + i];
          auto it = session->index.find(KeyId(want.key));
          if (it == session->index.end()) {
            available = false;
            break;
          }
          if (!SameRecord(want, it->second->record))
            return ErrorStatus(KVC_INCOMPATIBLE_MODEL, "catalog record disagrees with storage");
          resolved.push_back(it->second);
        }
      }
      if (available && (!best || candidate.prefix.token_count > best->prefix.token_count)) {
        best = &candidate;
        best_blocks = std::move(resolved);
      }
    }
    if (!best) return OkStatus();
    auto* read = new (std::nothrow) Read();
    if (!read) return ErrorStatus(KVC_RESOURCE_EXHAUSTED, "read handle allocation failed");
    read->blocks = std::move(best_blocks);
    for (const auto& block : read->blocks) read->manifest.push_back(block->record);
    read->header.kind = KVC_READ;
    read->header.session = session;
    session->objects_.fetch_add(1, std::memory_order_relaxed);
    out->prefix = best->prefix;
    out->read = reinterpret_cast<KvcObject*>(read);
    return OkStatus();
  });
}

KvcStatus KVC_CALL BeginWrite(KvcObject* object, const KvcWriteRequest* request, uint64_t,
                              KvcObject** out) {
  return Guard([&]() -> KvcStatus {
    auto* session = reinterpret_cast<Session*>(object);
    if (!request || request->struct_size < sizeof(KvcWriteRequest) || !out)
      return ErrorStatus(KVC_INVALID_ARGUMENT, "write request is malformed");
    *out = nullptr;
    std::lock_guard<std::mutex> lock(session->mutex);
    if (session->closing) return ErrorStatus(KVC_SHUTTING_DOWN, "session is closing");
    if (!session->configured) return ErrorStatus(KVC_UNAVAILABLE, "session is not configured");
    const uint64_t id = TxId(request->transaction);
    if (session->transactions.count(id))
      return ErrorStatus(KVC_CONFLICT, "transaction id is already in use");
    std::unordered_set<uint64_t> keys;
    uint64_t bytes = 0;
    auto tx = std::make_shared<TxRecord>();
    tx->id = request->transaction;
    for (uint64_t i = 0; i < request->block_count; ++i) {
      KvcStatus error;
      const Session::GroupCopy* group = ValidateBlock(session, request->blocks[i], &error);
      if (!group) return error;
      if (!keys.insert(KeyId(request->blocks[i].key)).second)
        return ErrorStatus(KVC_INVALID_ARGUMENT, "manifest keys must be unique");
      if (session->index.count(KeyId(request->blocks[i].key)) ||
          session->reserved_keys.count(KeyId(request->blocks[i].key)))
        return ErrorStatus(KVC_ALREADY_EXISTS, "block key is already published or reserved");
      bytes += group->payload_bytes;
      tx->manifest.push_back(request->blocks[i]);
      tx->covered.emplace_back(
          static_cast<size_t>(group->group.layer_count) * group->group.component_count, false);
    }
    if (session->bytes_used + bytes > kBudgetBytes)
      return ErrorStatus(KVC_RESOURCE_EXHAUSTED, "payload budget exceeded");
    tx->bytes = bytes;
    for (const auto& record : tx->manifest) {
      auto block = std::make_shared<Block>();
      block->record = record;
      block->payload.assign(FindGroup(session, record.group)->payload_bytes, 0);
      tx->blocks.push_back(std::move(block));
    }
    auto* write = new (std::nothrow) Write();
    if (!write) return ErrorStatus(KVC_RESOURCE_EXHAUSTED, "write handle allocation failed");
    write->tx = std::move(tx);
    write->header.kind = KVC_WRITE;
    write->header.session = session;
    for (const auto& block : write->tx->blocks)
      session->reserved_keys.insert(KeyId(block->record.key));
    session->bytes_used += write->tx->bytes;
    session->transactions.emplace(id, write->tx);
    session->objects_.fetch_add(1, std::memory_order_relaxed);
    *out = reinterpret_cast<KvcObject*>(write);
    return OkStatus();
  });
}

KvcStatus FillWriteStatus(const TxPtr& tx, KvcWriteStatus* out) {
  out->state = tx->state;
  out->blocks = tx->manifest.empty() ? nullptr : tx->manifest.data();
  out->block_count = tx->manifest.size();
  return OkStatus();
}

KvcStatus KVC_CALL Commit(KvcObject* object, KvcObject* write_handle, uint64_t,
                          KvcWriteStatus* out) {
  return Guard([&]() -> KvcStatus {
    auto* session = reinterpret_cast<Session*>(object);
    if (!out) return ErrorStatus(KVC_INVALID_ARGUMENT, "output pointer is null");
    out->state = 0;
    out->blocks = nullptr;
    out->block_count = 0;
    if (!Owns(write_handle, KVC_WRITE) || AsHeader(write_handle)->session != session)
      return ErrorStatus(KVC_STALE_HANDLE, "write handle is invalid for this session");
    auto* write = reinterpret_cast<Write*>(write_handle);
    std::lock_guard<std::mutex> lock(session->mutex);
    if (write->tx->state == KVC_WRITE_ABORTED)
      return ErrorStatus(KVC_ABORTED, "transaction was aborted");
    if (write->tx->state == KVC_COMMITTED) return FillWriteStatus(write->tx, out);
    for (const auto& covered : write->tx->covered)
      for (bool bit : covered)
        if (!bit) return ErrorStatus(KVC_INCOMPLETE, "coverage of the manifest is missing");
    if (!write->tx->transfers.empty())
      return ErrorStatus(KVC_BUSY, "stores of this transaction are still pending");
    for (const auto& block : write->tx->blocks) {
      session->reserved_keys.erase(KeyId(block->record.key));
      session->index.emplace(KeyId(block->record.key), block);
    }
    write->tx->state = KVC_COMMITTED;
    return FillWriteStatus(write->tx, out);
  });
}

KvcStatus KVC_CALL Abort(KvcObject* object, const KvcTransactionId* id, uint64_t) {
  return Guard([&]() -> KvcStatus {
    auto* session = reinterpret_cast<Session*>(object);
    if (!id) return ErrorStatus(KVC_INVALID_ARGUMENT, "transaction id is null");
    std::lock_guard<std::mutex> lock(session->mutex);
    auto it = session->transactions.find(TxId(*id));
    if (it == session->transactions.end())
      return ErrorStatus(KVC_NOT_FOUND, "transaction is unknown");
    if (it->second->state == KVC_COMMITTED)
      return ErrorStatus(KVC_ALREADY_COMMITTED, "transaction is committed");
    AbortLocked(*session, it->second);
    return OkStatus();
  });
}

KvcStatus KVC_CALL QueryWrite(KvcObject* object, const KvcTransactionId* id, uint64_t,
                              KvcWriteStatus* out) {
  return Guard([&]() -> KvcStatus {
    auto* session = reinterpret_cast<Session*>(object);
    if (!id || !out) return ErrorStatus(KVC_INVALID_ARGUMENT, "query is malformed");
    out->state = 0;
    out->blocks = nullptr;
    out->block_count = 0;
    std::lock_guard<std::mutex> lock(session->mutex);
    auto it = session->transactions.find(TxId(*id));
    if (it == session->transactions.end())
      return ErrorStatus(KVC_NOT_FOUND, "transaction is unknown");
    return FillWriteStatus(it->second, out);
  });
}

KvcStatus KVC_CALL Remove(KvcObject* object, const KvcDigest* key, uint64_t) {
  return Guard([&]() -> KvcStatus {
    auto* session = reinterpret_cast<Session*>(object);
    if (!key) return ErrorStatus(KVC_INVALID_ARGUMENT, "block key is null");
    std::lock_guard<std::mutex> lock(session->mutex);
    if (session->closing) return ErrorStatus(KVC_SHUTTING_DOWN, "session is closing");
    session->index.erase(KeyId(*key));
    return OkStatus();
  });
}

// ---------------------------------------------------------------------------
// Object table
// ---------------------------------------------------------------------------

KvcStatus KVC_CALL Manifest(const KvcObject* object, const KvcBlock** blocks, uint64_t* count) {
  return Guard([&]() -> KvcStatus {
    if (!object || !blocks || !count)
      return ErrorStatus(KVC_INVALID_ARGUMENT, "manifest query is malformed");
    const auto* header = reinterpret_cast<const Header*>(object);
    if (header->magic != kMagic || !header->session)
      return ErrorStatus(KVC_STALE_HANDLE, "object does not belong to this provider");
    if (header->kind == KVC_READ) {
      const auto* read = reinterpret_cast<const Read*>(object);
      *blocks = read->manifest.data();
      *count = read->manifest.size();
      return OkStatus();
    }
    if (header->kind == KVC_WRITE) {
      const auto* write = reinterpret_cast<const Write*>(object);
      std::lock_guard<std::mutex> lock(header->session->mutex);
      *blocks = write->tx->manifest.data();
      *count = write->tx->manifest.size();
      return OkStatus();
    }
    return ErrorStatus(KVC_STALE_HANDLE, "object has no block manifest");
  });
}

KvcStatus KVC_CALL MemoryInfo(const KvcObject* object, const KvcMemoryInfo** info) {
  return Guard([&]() -> KvcStatus {
    if (!object || !info) return ErrorStatus(KVC_INVALID_ARGUMENT, "query is malformed");
    const auto* header = reinterpret_cast<const Header*>(object);
    if (header->magic != kMagic || header->kind != KVC_MEMORY)
      return ErrorStatus(KVC_STALE_HANDLE, "object is not an imported region");
    *info = &reinterpret_cast<const Region*>(object)->info;
    return OkStatus();
  });
}

void KVC_CALL Abandon(KvcObject* object) {
  if (!Owns(object, KVC_WRITE)) return;
  auto* write = reinterpret_cast<Write*>(object);
  std::lock_guard<std::mutex> lock(write->header.session->mutex);
  AbortLocked(*write->header.session, write->tx);
}

// ---------------------------------------------------------------------------
// Data plane
// ---------------------------------------------------------------------------

bool StringIs(const KvcString& text, const char* literal) {
  const size_t length = std::strlen(literal);
  return text.size == length && (length == 0 || std::memcmp(text.data, literal, length) == 0);
}

KvcStatus KVC_CALL Import(KvcObject* object, const KvcMemoryImport* request, KvcObject** out) {
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
    session->objects_.fetch_add(1, std::memory_order_relaxed);
    *out = reinterpret_cast<KvcObject*>(region);
    return OkStatus();
  });
}

// Resolves and validates bindings; on success the caller receives fixed
// copy descriptors. Same rules as the host reference provider.
struct ResolvedBinding {
  size_t block_position;
  uint64_t payload_offset;
  unsigned char* host_address;
  uint64_t byte_count;
  size_t owned;
  size_t bit;
};

KvcStatus ResolveBindings(Session* session, const KvcSpan& span, const KvcBinding* bindings,
                          uint64_t binding_count, const std::vector<BlockPtr>& owned,
                          uint32_t required_access, bool writing, std::vector<BlockPtr>* blocks,
                          std::vector<ResolvedBinding>* entries,
                          std::vector<const KvcBinding*>* sources) {
  const Session::GroupCopy* group = FindGroup(session, span.group);
  if (!group) return ErrorStatus(KVC_INVALID_ARGUMENT, "span references an unknown group");
  if (!span.layers || span.layer_count == 0 || !span.blocks || span.block_count == 0)
    return ErrorStatus(KVC_INVALID_ARGUMENT, "span selection is empty");
  if (binding_count && !bindings)
    return ErrorStatus(KVC_INVALID_ARGUMENT, "bindings are missing");
  std::vector<uint32_t> layers(span.layers, span.layers + span.layer_count);
  for (size_t i = 0; i < layers.size(); ++i) {
    bool member = false;
    for (uint32_t layer : group->layers) member = member || layer == layers[i];
    if (!member) return ErrorStatus(KVC_INVALID_RANGE, "selected layer is not in the group");
    for (size_t j = 0; j < i; ++j)
      if (layers[j] == layers[i])
        return ErrorStatus(KVC_INVALID_ARGUMENT, "selected layers must be unique");
  }
  std::unordered_map<uint64_t, size_t> positions;
  for (size_t i = 0; i < owned.size(); ++i) positions.emplace(KeyId(owned[i]->record.key), i);
  std::unordered_map<uint64_t, size_t> selected;
  for (uint64_t b = 0; b < span.block_count; ++b) {
    const KvcSlice& slice = span.blocks[b];
    auto known = positions.find(KeyId(slice.key));
    if (known == positions.end())
      return ErrorStatus(KVC_INVALID_RANGE, "span block is not in the handle manifest");
    if (slice.first_token != 0 || slice.token_count != owned[known->second]->record.token_count)
      return ErrorStatus(KVC_UNSUPPORTED, "opaque components transfer whole blocks only");
    if (!selected.emplace(KeyId(slice.key), blocks->size()).second)
      return ErrorStatus(KVC_INVALID_ARGUMENT, "span block slices overlap");
    blocks->push_back(owned[known->second]);
  }
  const uint64_t expected = span.block_count * layers.size() * group->group.component_count;
  if (binding_count != expected)
    return ErrorStatus(KVC_INVALID_ARGUMENT, "bindings do not exactly cover the selection");

  const uint64_t row = group->payload_bytes / group->group.layer_count;
  for (uint64_t i = 0; i < binding_count; ++i) {
    const KvcBinding& binding = bindings[i];
    if (binding.layout != KVC_OPAQUE_BYTES)
      return ErrorStatus(KVC_UNSUPPORTED, "only opaque-byte bindings are supported");
    auto slot = selected.find(KeyId(binding.block.key));
    if (slot == selected.end())
      return ErrorStatus(KVC_INVALID_RANGE, "binding block is not in the span");
    bool layer_ok = false;
    for (uint32_t layer : layers) layer_ok = layer_ok || layer == binding.layer;
    if (!layer_ok) return ErrorStatus(KVC_INVALID_RANGE, "binding layer is not selected");
    if (binding.block.first_token != 0 ||
        binding.block.token_count != (*blocks)[slot->second]->record.token_count)
      return ErrorStatus(KVC_INVALID_RANGE, "binding extent differs from the span slice");
    size_t layer_at = group->layers.size(), component_at = group->components.size();
    for (size_t l = 0; l < group->layers.size(); ++l)
      if (group->layers[l] == binding.layer) layer_at = l;
    for (size_t c = 0; c < group->components.size(); ++c)
      if (group->components[c].id == binding.component) component_at = c;
    if (layer_at == group->layers.size() || component_at == group->components.size())
      return ErrorStatus(KVC_INVALID_RANGE, "layer or component is not in the group");
    if (!Owns(binding.region, KVC_MEMORY) || AsHeader(binding.region)->session != session)
      return ErrorStatus(KVC_STALE_HANDLE, "binding region is invalid for this session");
    auto* region = reinterpret_cast<Region*>(binding.region);
    if (binding.byte_count != group->components[component_at].bytes_per_block)
      return ErrorStatus(KVC_INVALID_RANGE, "binding must cover the whole component");
    if (required_access != 0 && region->info.access != required_access)
      return ErrorStatus(KVC_INVALID_ARGUMENT, "region access mode is insufficient");
    if (binding.byte_offset > region->info.byte_size ||
        binding.byte_count > region->info.byte_size - binding.byte_offset)
      return ErrorStatus(KVC_INVALID_RANGE, "binding exceeds its region");
    const size_t bit = layer_at * group->group.component_count + component_at;
    const size_t owned_position = positions.at(KeyId(binding.block.key));
    for (const auto& prior : *entries)
      if (prior.block_position == slot->second && prior.bit == bit)
        return writing ? ErrorStatus(KVC_CONFLICT, "overlapping write of one slot")
                       : ErrorStatus(KVC_INVALID_ARGUMENT, "duplicate binding of one slot");
    entries->push_back(
        {slot->second, layer_at * row + group->slot_offsets[component_at],
         reinterpret_cast<unsigned char*>(region->info.local_address) + binding.byte_offset,
         binding.byte_count, owned_position, bit});
    sources->push_back(&binding);
  }
  return OkStatus();
}

KvcStatus KVC_CALL Load(KvcObject* object, KvcObject* read_handle,
                        const KvcTransferRequest* request, KvcObject** out) {
  return Guard([&]() -> KvcStatus {
    auto* session = reinterpret_cast<Session*>(object);
    *out = nullptr;
    if (!request || request->struct_size < sizeof(KvcTransferRequest))
      return ErrorStatus(KVC_INVALID_ARGUMENT, "transfer request is malformed");
    if (!Owns(read_handle, KVC_READ) || AsHeader(read_handle)->session != session)
      return ErrorStatus(KVC_STALE_HANDLE, "read handle is invalid for this session");
    auto* read = reinterpret_cast<Read*>(read_handle);
    std::lock_guard<std::mutex> lock(session->mutex);
    if (session->closing) return ErrorStatus(KVC_SHUTTING_DOWN, "session is closing");
    for (uint64_t i = 0; i < request->dependency_count; ++i) {
      if (!Owns(request->dependencies[i], KVC_TRANSFER) ||
          AsHeader(request->dependencies[i])->session != session)
        return ErrorStatus(KVC_STALE_HANDLE, "dependency is not a transfer of this session");
      auto* dep = reinterpret_cast<Transfer*>(request->dependencies[i]);
      const uint32_t state = dep->state.load(std::memory_order_acquire);
      // Pending dependencies are accepted: the worker awaits them before
      // touching payloads. Already-failed dependencies reject the call.
      if (state != KVC_PENDING && state != KVC_SUCCEEDED)
        return ErrorStatus(KVC_ABORTED, "dependency did not complete successfully");
    }
    std::vector<BlockPtr> blocks;
    std::vector<ResolvedBinding> entries;
    std::vector<const KvcBinding*> sources;
    if (KvcStatus status = ResolveBindings(session, request->selection, request->bindings,
                                           request->binding_count, read->blocks, KVC_READ_WRITE,
                                           false, &blocks, &entries, &sources);
        status.code != KVC_OK)
      return status;

    auto* transfer = new (std::nothrow) Transfer();
    if (!transfer) return ErrorStatus(KVC_RESOURCE_EXHAUSTED, "transfer allocation failed");
    transfer->header.kind = KVC_TRANSFER;
    transfer->header.session = session;
    transfer->blocks = std::move(blocks);
    for (const auto& entry : entries)
      transfer->copies.push_back(
          {entry.host_address,
           transfer->blocks[entry.block_position]->payload.data() + entry.payload_offset,
           entry.byte_count});
    for (uint64_t i = 0; i < request->binding_count; ++i) Retain(request->bindings[i].region);
    for (uint64_t i = 0; i < request->binding_count; ++i)
      transfer->regions.push_back(AsHeader(request->bindings[i].region));
    for (uint64_t i = 0; i < request->dependency_count; ++i) {
      Retain(request->dependencies[i]);
      transfer->deps.push_back(AsHeader(request->dependencies[i]));
    }
    session->objects_.fetch_add(1, std::memory_order_relaxed);

    auto* item = new (std::nothrow) WorkItem(transfer);
    if (!item) {
      Release(reinterpret_cast<KvcObject*>(transfer));
      return ErrorStatus(KVC_RESOURCE_EXHAUSTED, "work item allocation failed");
    }
    Retain(reinterpret_cast<KvcObject*>(transfer));  // The queue's reference.
    if (!session->worker_started) {
      session->worker = std::thread(WorkerLoop, session);
      session->worker_started = true;
    }
    session->queue.push_back(item);
    session->work_signal.notify_one();
    *out = reinterpret_cast<KvcObject*>(transfer);
    return OkStatus();
  });
}

KvcStatus KVC_CALL Store(KvcObject* object, KvcObject* write_handle,
                         const KvcTransferRequest* request, KvcObject** out) {
  return Guard([&]() -> KvcStatus {
    auto* session = reinterpret_cast<Session*>(object);
    *out = nullptr;
    if (!request || request->struct_size < sizeof(KvcTransferRequest))
      return ErrorStatus(KVC_INVALID_ARGUMENT, "transfer request is malformed");
    if (!Owns(write_handle, KVC_WRITE) || AsHeader(write_handle)->session != session)
      return ErrorStatus(KVC_STALE_HANDLE, "write handle is invalid for this session");
    auto* write = reinterpret_cast<Write*>(write_handle);
    std::lock_guard<std::mutex> lock(session->mutex);
    if (session->closing) return ErrorStatus(KVC_SHUTTING_DOWN, "session is closing");
    if (write->tx->state == KVC_WRITE_ABORTED)
      return ErrorStatus(KVC_ABORTED, "transaction was aborted");
    if (write->tx->state == KVC_COMMITTED)
      return ErrorStatus(KVC_ALREADY_COMMITTED, "transaction is already committed");
    for (uint64_t i = 0; i < request->dependency_count; ++i) {
      if (!Owns(request->dependencies[i], KVC_TRANSFER) ||
          AsHeader(request->dependencies[i])->session != session)
        return ErrorStatus(KVC_STALE_HANDLE, "dependency is not a transfer of this session");
      auto* dep = reinterpret_cast<Transfer*>(request->dependencies[i]);
      const uint32_t state = dep->state.load(std::memory_order_acquire);
      // Pending dependencies are accepted: the worker awaits them before
      // touching payloads. Already-failed dependencies reject the call.
      if (state != KVC_PENDING && state != KVC_SUCCEEDED)
        return ErrorStatus(KVC_ABORTED, "dependency did not complete successfully");
    }
    std::vector<BlockPtr> blocks;
    std::vector<ResolvedBinding> entries;
    std::vector<const KvcBinding*> sources;
    if (KvcStatus status = ResolveBindings(session, request->selection, request->bindings,
                                           request->binding_count, write->tx->blocks, 0, true,
                                           &blocks, &entries, &sources);
        status.code != KVC_OK)
      return status;
    // Cross-request overlap and coverage application.
    for (const auto& entry : entries)
      if (write->tx->covered.at(entry.owned)[entry.bit])
        return ErrorStatus(KVC_CONFLICT, "overlapping write of one slot");

    auto* transfer = new (std::nothrow) Transfer();
    if (!transfer) return ErrorStatus(KVC_RESOURCE_EXHAUSTED, "transfer allocation failed");
    transfer->header.kind = KVC_TRANSFER;
    transfer->header.session = session;
    transfer->store = true;
    transfer->tx = write->tx;
    transfer->blocks = std::move(blocks);
    for (const auto& entry : entries)
      transfer->copies.push_back(
          {transfer->blocks[entry.block_position]->payload.data() + entry.payload_offset,
           entry.host_address, entry.byte_count});
    for (uint64_t i = 0; i < request->binding_count; ++i) {
      Retain(request->bindings[i].region);
      transfer->regions.push_back(AsHeader(request->bindings[i].region));
    }
    for (uint64_t i = 0; i < request->dependency_count; ++i) {
      Retain(request->dependencies[i]);
      transfer->deps.push_back(AsHeader(request->dependencies[i]));
    }
    const uint64_t store_index = ++session->accepted_stores;
    transfer->fail = session->fail_store != 0 && store_index == session->fail_store;
    for (const auto& entry : entries) write->tx->covered[entry.owned][entry.bit] = true;
    if (write->tx->state == KVC_ALLOCATED) write->tx->state = KVC_WRITING;
    write->tx->transfers.push_back(&transfer->header);
    session->objects_.fetch_add(1, std::memory_order_relaxed);

    auto* item = new (std::nothrow) WorkItem(transfer);
    if (!item) {
      Release(reinterpret_cast<KvcObject*>(transfer));
      return ErrorStatus(KVC_RESOURCE_EXHAUSTED, "work item allocation failed");
    }
    Retain(reinterpret_cast<KvcObject*>(transfer));
    if (!session->worker_started) {
      session->worker = std::thread(WorkerLoop, session);
      session->worker_started = true;
    }
    session->queue.push_back(item);
    session->work_signal.notify_one();
    *out = reinterpret_cast<KvcObject*>(transfer);
    return OkStatus();
  });
}

// ---------------------------------------------------------------------------
// Transfer table
// ---------------------------------------------------------------------------

KvcStatus KVC_CALL Poll(KvcObject* object, KvcTransferStatus* out) {
  return Guard([&]() -> KvcStatus {
    if (!out || out->struct_size < sizeof(*out))
      return ErrorStatus(KVC_INVALID_ARGUMENT, "poll record is too small");
    if (!Owns(object, KVC_TRANSFER))
      return ErrorStatus(KVC_STALE_HANDLE, "object is not a transfer");
    auto* transfer = reinterpret_cast<Transfer*>(object);
    const uint32_t state = transfer->state.load(std::memory_order_acquire);
    out->state = state;
    out->result = transfer->final_code == KVC_OK
                      ? KvcStatus{KVC_OK, {}, nullptr, nullptr}
                      : ErrorStatus(transfer->final_code, transfer->final_text);
    return OkStatus();
  });
}

KvcStatus KVC_CALL Wait(KvcObject* object, uint64_t timeout) {
  return Guard([&]() -> KvcStatus {
    if (!Owns(object, KVC_TRANSFER))
      return ErrorStatus(KVC_STALE_HANDLE, "object is not a transfer");
    auto* transfer = reinterpret_cast<Transfer*>(object);
    auto* session = AsHeader(object)->session;
    const auto deadline = std::chrono::steady_clock::now() +
                          std::chrono::nanoseconds(static_cast<int64_t>(timeout));
    std::unique_lock<std::mutex> lock(session->mutex);
    while (transfer->state.load(std::memory_order_acquire) == KVC_PENDING) {
      if (timeout != KVC_TIMEOUT_INFINITE && std::chrono::steady_clock::now() >= deadline)
        return ErrorStatus(KVC_DEADLINE_EXCEEDED, "transfer is still pending");
      session->work_signal.wait_for(lock, std::chrono::milliseconds(2));
    }
    if (transfer->state.load(std::memory_order_acquire) == KVC_SUCCEEDED) return OkStatus();
    return ErrorStatus(transfer->final_code, transfer->final_text);
  });
}

// ---------------------------------------------------------------------------
// Tables and entry point
// ---------------------------------------------------------------------------

#define TABLE_HEADER(T) sizeof(T), KVC_ABI_MAJOR, KVC_ABI_MINOR

const KvcProviderApi provider_api = {
    TABLE_HEADER(KvcProviderApi), Capabilities, Configure, Configuration, Shutdown, nullptr};
const KvcControl control_api = {
    TABLE_HEADER(KvcControl), Lookup, BeginWrite, Commit, Abort, QueryWrite, Remove};
const KvcData data_api = {TABLE_HEADER(KvcData), Import, Load, Store, nullptr};
const KvcObjectApi object_api = {
    TABLE_HEADER(KvcObjectApi), Retain, Release, Kind, Manifest, MemoryInfo, Abandon};
const KvcTransferApi transfer_api = {TABLE_HEADER(KvcTransferApi), Poll, Wait, nullptr};
const KvcPlugin plugin = {TABLE_HEADER(KvcPlugin),
                          {"async.fixture", 13},
                          Open,
                          &provider_api,
                          &control_api,
                          &data_api,
                          &object_api,
                          &transfer_api};

#undef TABLE_HEADER

}  // namespace

KVC_EXPORT KvcStatus KVC_CALL KvcGetProvider(uint32_t requested_major, uint32_t requested_minor,
                                             const KvcPlugin** out) {
  (void)requested_minor;
  if (!out) return KvcStatus{KVC_INVALID_ARGUMENT, {}, nullptr, nullptr};
  *out = nullptr;
  if (requested_major != KVC_ABI_MAJOR) return KvcStatus{KVC_UNSUPPORTED, {}, nullptr, nullptr};
  *out = &plugin;
  return KvcStatus{KVC_OK, {}, nullptr, nullptr};
}
