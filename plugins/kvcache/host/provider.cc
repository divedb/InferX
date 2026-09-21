// Session lifecycle, negotiation, object table, and the control plane of
// the host reference provider.

#include <cstring>
#include <new>
#include <utility>

#include "internal.h"

namespace kvc_host {
namespace {

KvcStatus FillWriteStatus(const TxPtr& tx, KvcWriteStatus* out) {
  out->state = tx->state;
  out->blocks = tx->manifest.empty() ? nullptr : tx->manifest.data();
  out->block_count = tx->manifest.size();
  return OkStatus();
}

bool SchemaIs(const KvcSchema& schema, const char* name, uint32_t major) {
  const size_t length = std::strlen(name);
  return schema.major == major && schema.name.size == length &&
         (length == 0 || std::memcmp(schema.name.data, name, length) == 0);
}

bool SameRecord(const KvcBlock& a, const KvcBlock& b) {
  return a.group == b.group && a.logical_index == b.logical_index &&
         a.first_token == b.first_token && a.token_count == b.token_count &&
         std::memcmp(a.dependency.bytes, b.dependency.bytes, sizeof(a.dependency.bytes)) == 0;
}

// Frees storage of an unfinished transaction. The session mutex is held.
void AbortLocked(Session& session, const TxPtr& tx) {
  if (tx->state == KVC_COMMITTED || tx->state == KVC_WRITE_ABORTED) return;
  for (const auto& block : tx->blocks)
    session.reserved_keys.erase(Session::KeyId(block->record.key));
  tx->state = KVC_WRITE_ABORTED;
  tx->covered.clear();
  tx->blocks.clear();  // Payload destructors return the reserved bytes.
}

}  // namespace

Block::~Block() {
  // The shutdown protocol guarantees the session outlives every block.
  if (!payload.empty())
    session->bytes_used.fetch_sub(payload.size(), std::memory_order_relaxed);
}

KvcStatus OkStatus() { return KvcStatus{KVC_OK, {}, nullptr, nullptr}; }

KvcStatus ErrorStatus(uint32_t code, std::string diagnostic) {
  auto* text = new (std::nothrow) std::string(std::move(diagnostic));
  if (!text) return KvcStatus{code, {}, nullptr, nullptr};
  return KvcStatus{
      code, {reinterpret_cast<const char*>(text->data()), text->size()}, text, [](void* owner) {
        delete static_cast<std::string*>(owner);
      }};
}

uint64_t Session::KeyId(const KvcDigest& key) {
  uint64_t view = 0;
  std::memcpy(&view, key.bytes, sizeof(view));
  return view;
}

uint64_t Session::TxId(const KvcTransactionId& id) {
  uint64_t view = 0;
  std::memcpy(&view, id.bytes, sizeof(view));
  return view;
}

const Session::GroupCopy* Session::FindGroup(uint32_t id) const {
  for (const auto& group : groups)
    if (group.group.id == id) return &group;
  return nullptr;
}

const Session::GroupCopy* Session::ValidateBlock(const KvcBlock& block,
                                                 KvcStatus* error) const {
  *error = OkStatus();
  const GroupCopy* group = FindGroup(block.group);
  if (!group) {
    *error = ErrorStatus(KVC_INVALID_ARGUMENT, "block references an unknown group");
    return nullptr;
  }
  const uint32_t tokens = group->group.tokens_per_block;
  if (block.logical_index > UINT64_MAX / tokens) {
    *error = ErrorStatus(KVC_INVALID_RANGE, "block logical index overflows");
    return nullptr;
  }
  if (block.first_token != block.logical_index * tokens || block.token_count == 0 ||
      block.token_count > tokens) {
    *error = ErrorStatus(KVC_INVALID_RANGE, "block token extent is inconsistent");
    return nullptr;
  }
  return group;
}

Session::~Session() = default;

// ---------------------------------------------------------------------------
// Session lifetime and negotiation
// ---------------------------------------------------------------------------

KvcStatus OpenSession(const KvcDescriptor* options, uint64_t timeout, KvcObject** out) {
  (void)options;
  (void)timeout;
  return Guard([&]() -> KvcStatus {
    if (!out) return ErrorStatus(KVC_INVALID_ARGUMENT, "output pointer is null");
    auto* session = new (std::nothrow) Session();
    if (!session) return ErrorStatus(KVC_RESOURCE_EXHAUSTED, "session allocation failed");
    session->header.kind = KVC_SESSION;
    session->header.session = session;
    *out = reinterpret_cast<KvcObject*>(session);
    return OkStatus();
  });
}

void SessionRetain(KvcObject* object) {
  Session::AsHeader(object)->refs.fetch_add(1, std::memory_order_relaxed);
}

void SessionRelease(KvcObject* object) {
  auto* session = reinterpret_cast<Session*>(object);
  if (session->header.refs.fetch_sub(1, std::memory_order_acq_rel) == 1) delete session;
}

KvcStatus SessionCapabilities(KvcObject* object, KvcCapabilities* out) {
  (void)object;
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

KvcStatus SessionConfigure(KvcObject* object, const KvcConfig* request, uint64_t timeout) {
  (void)timeout;
  return Guard([&]() -> KvcStatus {
    auto* session = reinterpret_cast<Session*>(object);
    if (!request || request->struct_size < sizeof(KvcConfig))
      return ErrorStatus(KVC_INVALID_ARGUMENT, "configuration record is too small");
    std::lock_guard<std::mutex> lock(session->mutex);
    if (session->configured)
      return ErrorStatus(KVC_INVALID_ARGUMENT, "session is already configured");
    if (session->closing) return ErrorStatus(KVC_SHUTTING_DOWN, "session is closing");
    if (request->required_features & ~static_cast<uint64_t>(KVC_FEATURE_LAYERWISE))
      return ErrorStatus(KVC_UNSUPPORTED, "a required feature is not supported");
    if (request->required_extension_count)
      return ErrorStatus(KVC_UNSUPPORTED, "no extensions are supported");
    if (!request->groups || request->group_count == 0)
      return ErrorStatus(KVC_INVALID_ARGUMENT, "at least one group is required");

    // Build and validate a full deep copy; install it only on success. The
    // borrowable KvcConfig pointers are fixed after the vectors settle.
    std::vector<Session::GroupCopy> built;
    built.reserve(request->group_count);
    std::vector<uint32_t> seen_groups;
    for (uint64_t g = 0; g < request->group_count; ++g) {
      const KvcGroup& group = request->groups[g];
      if (!group.layers || group.layer_count == 0 || !group.components ||
          group.component_count == 0)
        return ErrorStatus(KVC_INVALID_ARGUMENT, "a group needs layers and components");
      if (group.tokens_per_block == 0)
        return ErrorStatus(KVC_INVALID_ARGUMENT, "tokens_per_block must be positive");
      if (!SchemaIs(group.semantics.schema, "kvc.opaque", 1))
        return ErrorStatus(KVC_UNSUPPORTED, "group schema must be kvc.opaque v1");
      for (uint32_t seen : seen_groups)
        if (seen == group.id)
          return ErrorStatus(KVC_INVALID_ARGUMENT, "group ids must be unique");
      seen_groups.push_back(group.id);

      Session::GroupCopy stored;
      stored.group.id = group.id;
      stored.group.tokens_per_block = group.tokens_per_block;
      stored.semantics_name.assign(group.semantics.schema.name.data,
                                   group.semantics.schema.name.size);
      stored.semantics_payload.assign(
          group.semantics.canonical_payload.data,
          group.semantics.canonical_payload.data + group.semantics.canonical_payload.size);
      stored.layers.assign(group.layers, group.layers + group.layer_count);
      for (size_t i = 0; i < stored.layers.size(); ++i)
        if (i && stored.layers[i - 1] >= stored.layers[i])
          return ErrorStatus(KVC_INVALID_ARGUMENT, "layers must be ascending and unique");

      uint64_t row_bytes = 0;
      for (uint64_t c = 0; c < group.component_count; ++c) {
        const KvcComponent& component = group.components[c];
        if (component.layout != KVC_OPAQUE_BYTES)
          return ErrorStatus(KVC_UNSUPPORTED, "component layout must be opaque bytes");
        if (!SchemaIs(component.meaning.schema, "kvc.bytes", 1))
          return ErrorStatus(KVC_UNSUPPORTED, "component schema must be kvc.bytes v1");
        if (component.meaning.canonical_payload.size != 0)
          return ErrorStatus(KVC_INVALID_ARGUMENT, "opaque components take no parameters");
        if (component.bytes_per_block == 0)
          return ErrorStatus(KVC_INVALID_ARGUMENT, "bytes_per_block must be positive");
        for (const auto& prior : stored.components)
          if (prior.id == component.id)
            return ErrorStatus(KVC_INVALID_ARGUMENT, "component ids must be unique");
        stored.slot_offsets.push_back(row_bytes);
        row_bytes += component.bytes_per_block;
        stored.components.push_back(component);
        stored.component_names.emplace_back(component.meaning.schema.name.data,
                                            component.meaning.schema.name.size);
      }
      if (row_bytes > kBudgetBytes)
        return ErrorStatus(KVC_UNSUPPORTED, "block payload size exceeds the budget");
      stored.payload_bytes = row_bytes * group.layer_count;

      // Fix component strings to point at owned storage before the vectors
      // move into the session (the caller's arrays are only borrowed).
      for (size_t c = 0; c < stored.components.size(); ++c) {
        stored.components[c].meaning.schema.name = {stored.component_names[c].data(),
                                                    stored.component_names[c].size()};
      }
      built.push_back(std::move(stored));
    }
    // Freeze the vectors, then install pointers that stay valid for the
    // session lifetime.
    session->groups = std::move(built);
    session->group_values.clear();

    for (auto& group : session->groups) {
      group.group.layers = group.layers.data();
      group.group.layer_count = group.layers.size();
      group.group.components = group.components.data();
      group.group.component_count = group.components.size();
      group.group.semantics.schema.name = {group.semantics_name.data(),
                                           group.semantics_name.size()};
      group.group.semantics.canonical_payload = {group.semantics_payload.data(),
                                                 group.semantics_payload.size()};
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

KvcStatus SessionConfiguration(KvcObject* object, const KvcConfig** out) {
  return Guard([&]() -> KvcStatus {
    auto* session = reinterpret_cast<Session*>(object);
    if (!out) return ErrorStatus(KVC_INVALID_ARGUMENT, "output pointer is null");
    std::lock_guard<std::mutex> lock(session->mutex);
    if (!session->configured) return ErrorStatus(KVC_UNAVAILABLE, "session is not configured");
    *out = &session->config;
    return OkStatus();
  });
}

KvcStatus SessionQueryExtension(KvcObject* object, const KvcSchema* schema, KvcExtension* out) {
  return Guard([&]() -> KvcStatus {
    if (!schema || !out) return ErrorStatus(KVC_INVALID_ARGUMENT, "query is malformed");
    if (schema->name.size != KVC_STATS_SCHEMA_NAME_LEN ||
        std::memcmp(schema->name.data, KVC_STATS_SCHEMA_NAME, KVC_STATS_SCHEMA_NAME_LEN) != 0 ||
        schema->major != KVC_STATS_SCHEMA_MAJOR)
      return ErrorStatus(KVC_UNSUPPORTED, "unknown extension schema");
    static const KvcStatsApi stats_api = {sizeof(KvcStatsApi), KVC_ABI_MAJOR, KVC_ABI_MINOR,
                                          SessionStats};
    out->struct_size = sizeof(*out);
    out->table = &stats_api;
    RetainObject(object);  // The caller releases this reference.
    out->owner = object;
    return OkStatus();
  });
}

KvcStatus SessionStats(KvcObject* object, KvcStatsRecord* out) {
  return Guard([&]() -> KvcStatus {
    auto* session = reinterpret_cast<Session*>(object);
    if (!out || out->struct_size < sizeof(*out))
      return ErrorStatus(KVC_INVALID_ARGUMENT, "stats record is too small");
    *out = KvcStatsRecord{};
    out->struct_size = sizeof(*out);
    out->lookups = session->stats_lookups.load(std::memory_order_relaxed);
    out->lookup_hits = session->stats_lookup_hits.load(std::memory_order_relaxed);
    out->lookup_hit_blocks = session->stats_lookup_hit_blocks.load(std::memory_order_relaxed);
    out->blocks_committed = session->stats_blocks_committed.load(std::memory_order_relaxed);
    out->blocks_retired = session->stats_blocks_retired.load(std::memory_order_relaxed);
    out->blocks_evicted = session->stats_blocks_evicted.load(std::memory_order_relaxed);
    out->transactions_begun = session->stats_tx_begun.load(std::memory_order_relaxed);
    out->transactions_committed = session->stats_tx_committed.load(std::memory_order_relaxed);
    out->transactions_aborted = session->stats_tx_aborted.load(std::memory_order_relaxed);
    out->stores = session->stats_stores.load(std::memory_order_relaxed);
    out->store_bytes = session->stats_store_bytes.load(std::memory_order_relaxed);
    out->loads = session->stats_loads.load(std::memory_order_relaxed);
    out->load_bytes = session->stats_load_bytes.load(std::memory_order_relaxed);
    // The host provider has a single CPU tier.
    out->host_bytes_in_use = session->bytes_used.load(std::memory_order_relaxed);
    out->host_budget = kBudgetBytes;
    return OkStatus();
  });
}

KvcStatus SessionShutdown(KvcObject* object, uint64_t timeout) {
  (void)timeout;
  return Guard([&]() -> KvcStatus {
    auto* session = reinterpret_cast<Session*>(object);
    std::lock_guard<std::mutex> lock(session->mutex);
    session->closing = true;
    if (session->objects_.load(std::memory_order_acquire) != 0)
      return ErrorStatus(KVC_BUSY, "session objects are still live");
    return OkStatus();
  });
}

// ---------------------------------------------------------------------------
// Control plane
// ---------------------------------------------------------------------------

KvcStatus Lookup(KvcObject* object, const KvcLookupRequest* request, uint64_t timeout,
                 KvcLookupResult* out) {
  (void)timeout;
  return Guard([&]() -> KvcStatus {
    auto* session = reinterpret_cast<Session*>(object);
    if (!request || request->struct_size < sizeof(KvcLookupRequest) || !out)
      return ErrorStatus(KVC_INVALID_ARGUMENT, "lookup request is malformed");
    if (request->catalog_count && !request->catalog)
      return ErrorStatus(KVC_INVALID_ARGUMENT, "catalog is missing");
    if (request->candidate_count && !request->candidates)
      return ErrorStatus(KVC_INVALID_ARGUMENT, "candidates are missing");
    out->read = nullptr;
    out->prefix = KvcPrefix{};
    std::lock_guard<std::mutex> lock(session->mutex);
    if (session->closing) return ErrorStatus(KVC_SHUTTING_DOWN, "session is closing");
    if (!session->configured) return ErrorStatus(KVC_UNAVAILABLE, "session is not configured");

    // Validate the catalog: unique keys with consistent extents.
    std::unordered_set<uint64_t> keys;
    keys.reserve(request->catalog_count);
    for (uint64_t i = 0; i < request->catalog_count; ++i) {
      KvcStatus error;
      if (!session->ValidateBlock(request->catalog[i], &error)) return error;
      if (!keys.insert(Session::KeyId(request->catalog[i].key)).second)
        return ErrorStatus(KVC_INVALID_ARGUMENT, "catalog keys must be unique");
    }

    // Evaluate every candidate independently; availability need not be
    // monotonic. The longest fully available candidate wins.
    const KvcCandidate* best = nullptr;
    std::vector<BlockPtr> best_blocks;
    for (uint64_t c = 0; c < request->candidate_count; ++c) {
      const KvcCandidate& candidate = request->candidates[c];
      if (candidate.required_count && !candidate.required)
        return ErrorStatus(KVC_INVALID_ARGUMENT, "candidate ranges are missing");
      std::vector<BlockPtr> resolved;
      bool available = true;
      for (uint64_t r = 0; r < candidate.required_count && available; ++r) {
        const KvcRange& range = candidate.required[r];
        if (range.first > request->catalog_count ||
            range.count > request->catalog_count - range.first)
          return ErrorStatus(KVC_INVALID_RANGE, "candidate range exceeds the catalog");
        for (uint64_t i = 0; i < range.count; ++i) {
          const KvcBlock& want = request->catalog[range.first + i];
          auto it = session->index.find(Session::KeyId(want.key));
          if (it == session->index.end()) {
            available = false;
            break;
          }
          if (!SameRecord(want, it->second->record))
            return ErrorStatus(KVC_INCOMPATIBLE_MODEL,
                               "catalog record disagrees with stored block metadata");
          resolved.push_back(it->second);
        }
      }
      if (!available) continue;
      if (!best || candidate.prefix.token_count > best->prefix.token_count) {
        best = &candidate;
        best_blocks = std::move(resolved);
      }
    }

    session->stats_lookups.fetch_add(1, std::memory_order_relaxed);
    if (!best) return OkStatus();  // A miss is a success with no read handle.
    session->stats_lookup_hits.fetch_add(1, std::memory_order_relaxed);
    session->stats_lookup_hit_blocks.fetch_add(best_blocks.size(), std::memory_order_relaxed);
    const uint64_t tick = session->lru_tick.fetch_add(1);
    for (auto& block : best_blocks) block->last_use = tick;
    auto* read = new (std::nothrow) Read();
    if (!read) return ErrorStatus(KVC_RESOURCE_EXHAUSTED, "read handle allocation failed");
    read->blocks = std::move(best_blocks);
    read->manifest.reserve(read->blocks.size());
    for (const auto& block : read->blocks) read->manifest.push_back(block->record);
    read->header.kind = KVC_READ;
    read->header.session = session;
    session->ObjectCreated();
    out->prefix = best->prefix;
    out->read = reinterpret_cast<KvcObject*>(read);
    return OkStatus();
  });
}

KvcStatus BeginWrite(KvcObject* object, const KvcWriteRequest* request, uint64_t timeout,
                     KvcObject** out) {
  (void)timeout;
  return Guard([&]() -> KvcStatus {
    auto* session = reinterpret_cast<Session*>(object);
    if (!request || request->struct_size < sizeof(KvcWriteRequest) || !out)
      return ErrorStatus(KVC_INVALID_ARGUMENT, "write request is malformed");
    if (request->block_count && !request->blocks)
      return ErrorStatus(KVC_INVALID_ARGUMENT, "manifest blocks are missing");
    *out = nullptr;
    std::lock_guard<std::mutex> lock(session->mutex);
    if (session->closing) return ErrorStatus(KVC_SHUTTING_DOWN, "session is closing");
    if (!session->configured) return ErrorStatus(KVC_UNAVAILABLE, "session is not configured");

    const uint64_t id = Session::TxId(request->transaction);
    if (session->transactions.count(id))
      return ErrorStatus(KVC_CONFLICT, "transaction id is already in use");

    // Validate the manifest and reserve budget. Payloads are allocated
    // eagerly so that stores cannot fail with resources mid-transaction.
    std::unordered_set<uint64_t> keys;
    uint64_t bytes = 0;
    TxPtr tx = std::make_shared<TxRecord>();
    tx->id = request->transaction;
    tx->manifest.reserve(request->block_count);
    for (uint64_t i = 0; i < request->block_count; ++i) {
      const KvcBlock& record = request->blocks[i];
      KvcStatus error;
      const Session::GroupCopy* group = session->ValidateBlock(record, &error);
      if (!group) return error;
      const uint64_t key = Session::KeyId(record.key);
      if (!keys.insert(key).second)
        return ErrorStatus(KVC_INVALID_ARGUMENT, "manifest keys must be unique");
      if (session->index.count(key) || session->reserved_keys.count(key))
        return ErrorStatus(KVC_ALREADY_EXISTS, "block key is already published or reserved");
      bytes += group->payload_bytes;
      tx->manifest.push_back(record);
      tx->covered.emplace_back(
          static_cast<size_t>(group->group.layer_count) * group->group.component_count, false);
    }
    if (session->bytes_used.load(std::memory_order_relaxed) + bytes > kBudgetBytes)
      return ErrorStatus(KVC_RESOURCE_EXHAUSTED, "payload budget exceeded");
    tx->bytes = bytes;
    for (const auto& record : tx->manifest) {
      auto block = std::make_shared<Block>();
      block->record = record;
      block->session = session;
      block->payload.assign(session->FindGroup(record.group)->payload_bytes, 0);
      tx->blocks.push_back(std::move(block));
    }
    auto* write = new (std::nothrow) Write();
    if (!write) return ErrorStatus(KVC_RESOURCE_EXHAUSTED, "write handle allocation failed");
    write->tx = std::move(tx);
    write->header.kind = KVC_WRITE;
    write->header.session = session;

    // Nothing below can throw: install the transaction.
    for (const auto& block : write->tx->blocks) {
      session->reserved_keys.insert(Session::KeyId(block->record.key));
      session->bytes_used.fetch_add(block->payload.size(), std::memory_order_relaxed);
    }
    session->transactions.emplace(id, write->tx);
    session->stats_tx_begun.fetch_add(1, std::memory_order_relaxed);
    session->ObjectCreated();
    *out = reinterpret_cast<KvcObject*>(write);
    return OkStatus();
  });
}

KvcStatus Commit(KvcObject* object, KvcObject* write_handle, uint64_t timeout,
                 KvcWriteStatus* out) {
  (void)timeout;
  return Guard([&]() -> KvcStatus {
    auto* session = reinterpret_cast<Session*>(object);
    if (!out) return ErrorStatus(KVC_INVALID_ARGUMENT, "output pointer is null");
    out->state = 0;
    out->blocks = nullptr;
    out->block_count = 0;
    if (!Session::Owns(write_handle, KVC_WRITE) ||
        Session::AsHeader(write_handle)->session != session)
      return ErrorStatus(KVC_STALE_HANDLE, "write handle is invalid for this session");
    auto* write = reinterpret_cast<Write*>(write_handle);
    std::lock_guard<std::mutex> lock(session->mutex);
    if (write->tx->state == KVC_WRITE_ABORTED)
      return ErrorStatus(KVC_ABORTED, "transaction was aborted");
    if (write->tx->state == KVC_COMMITTED) return FillWriteStatus(write->tx, out);

    for (const auto& covered : write->tx->covered)
      for (bool bit : covered)
        if (!bit) return ErrorStatus(KVC_INCOMPLETE, "coverage of the manifest is missing");
    // Atomically publish: reservations guarantee every key is still free.
    // The transaction keeps its record manifest for receipts but drops its
    // block references; the index owns published blocks from here on.
    for (const auto& block : write->tx->blocks) {
      session->reserved_keys.erase(Session::KeyId(block->record.key));
      session->index.emplace(Session::KeyId(block->record.key), block);
      block->last_use = session->lru_tick.fetch_add(1);
    }
    session->stats_tx_committed.fetch_add(1, std::memory_order_relaxed);
    session->stats_blocks_committed.fetch_add(write->tx->manifest.size(),
                                              std::memory_order_relaxed);
    write->tx->blocks.clear();
    write->tx->state = KVC_COMMITTED;
    return FillWriteStatus(write->tx, out);
  });
}

KvcStatus AbortWrite(KvcObject* object, const KvcTransactionId* id, uint64_t timeout) {
  (void)timeout;
  return Guard([&]() -> KvcStatus {
    auto* session = reinterpret_cast<Session*>(object);
    if (!id) return ErrorStatus(KVC_INVALID_ARGUMENT, "transaction id is null");
    std::lock_guard<std::mutex> lock(session->mutex);
    auto it = session->transactions.find(Session::TxId(*id));
    if (it == session->transactions.end())
      return ErrorStatus(KVC_NOT_FOUND, "transaction is unknown");
    if (it->second->state == KVC_COMMITTED)
      return ErrorStatus(KVC_ALREADY_COMMITTED, "transaction is committed");
    AbortLocked(*session, it->second);
    return OkStatus();
  });
}

KvcStatus QueryWrite(KvcObject* object, const KvcTransactionId* id, uint64_t timeout,
                     KvcWriteStatus* out) {
  (void)timeout;
  return Guard([&]() -> KvcStatus {
    auto* session = reinterpret_cast<Session*>(object);
    if (!id || !out) return ErrorStatus(KVC_INVALID_ARGUMENT, "query is malformed");
    out->state = 0;
    out->blocks = nullptr;
    out->block_count = 0;
    std::lock_guard<std::mutex> lock(session->mutex);
    auto it = session->transactions.find(Session::TxId(*id));
    if (it == session->transactions.end())
      return ErrorStatus(KVC_NOT_FOUND, "transaction is unknown");
    return FillWriteStatus(it->second, out);
  });
}

KvcStatus RemoveBlock(KvcObject* object, const KvcDigest* key, uint64_t timeout) {
  (void)timeout;
  return Guard([&]() -> KvcStatus {
    auto* session = reinterpret_cast<Session*>(object);
    if (!key) return ErrorStatus(KVC_INVALID_ARGUMENT, "block key is null");
    std::lock_guard<std::mutex> lock(session->mutex);
    if (session->closing) return ErrorStatus(KVC_SHUTTING_DOWN, "session is closing");
    // Idempotent retirement: pinned readers keep their generations.
    if (session->index.erase(Session::KeyId(*key)) != 0)
      session->stats_blocks_retired.fetch_add(1, std::memory_order_relaxed);
    return OkStatus();
  });
}

// ---------------------------------------------------------------------------
// Object table
// ---------------------------------------------------------------------------

void RetainObject(KvcObject* object) {
  if (object) Session::AsHeader(object)->refs.fetch_add(1, std::memory_order_relaxed);
}

void ReleaseObject(KvcObject* object) {
  if (!object) return;
  auto* header = Session::AsHeader(object);
  if (header->magic != kMagic) return;
  if (header->refs.fetch_sub(1, std::memory_order_acq_rel) != 1) return;
  Session* session = header->session;
  switch (header->kind) {
    case KVC_SESSION: {
      SessionRelease(reinterpret_cast<KvcObject*>(session));
      return;
    }
    case KVC_READ: {
      delete reinterpret_cast<Read*>(object);
      break;
    }
    case KVC_WRITE: {
      auto* write = reinterpret_cast<Write*>(object);
      if (write->tx) {
        std::lock_guard<std::mutex> lock(session->mutex);
        AbortLocked(*session, write->tx);
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
      std::vector<Header*> regions;
      std::vector<Header*> deps;
      {
        std::lock_guard<std::mutex> lock(transfer->mutex);
        regions = std::move(transfer->regions);
        deps = std::move(transfer->deps);
        transfer->blocks.clear();
      }
      delete transfer;
      for (Header* region : regions) ReleaseObject(reinterpret_cast<KvcObject*>(region));
      for (Header* dep : deps) ReleaseObject(reinterpret_cast<KvcObject*>(dep));
      break;
    }
    default:
      return;
  }
  session->ObjectDestroyed();
}

uint32_t ObjectKind(const KvcObject* object) {
  if (!object) return 0;
  const auto* header = reinterpret_cast<const Header*>(object);
  return header->magic == kMagic ? header->kind : 0;
}

KvcStatus ObjectManifest(const KvcObject* object, const KvcBlock** blocks, uint64_t* count) {
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

KvcStatus ObjectMemoryInfo(const KvcObject* object, const KvcMemoryInfo** info) {
  return Guard([&]() -> KvcStatus {
    if (!object || !info) return ErrorStatus(KVC_INVALID_ARGUMENT, "info query is malformed");
    const auto* header = reinterpret_cast<const Header*>(object);
    if (header->magic != kMagic || header->kind != KVC_MEMORY)
      return ErrorStatus(KVC_STALE_HANDLE, "object is not an imported region");
    *info = &reinterpret_cast<const Region*>(object)->info;
    return OkStatus();
  });
}

void AbandonWrite(KvcObject* object) {
  if (!Session::Owns(object, KVC_WRITE)) return;
  auto* write = reinterpret_cast<Write*>(object);
  Session* session = write->header.session;
  std::lock_guard<std::mutex> lock(session->mutex);
  AbortLocked(*session, write->tx);
}

}  // namespace kvc_host
