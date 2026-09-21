// Host reference provider: session-local CPU storage for the fixed
// opaque-component profile. Implements every required ABI operation with
// synchronous copies that return already-terminal transfer handles. See
// docs/extensions/kvcache.md for the supported profile.

#ifndef KVCACHE_HOST_INTERNAL_H_
#define KVCACHE_HOST_INTERNAL_H_

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "kvc/abi/provider.h"
#include "kvc/extensions/stats.h"

namespace kvc_host {

// Session payload budget, including unpublished writes and retired blocks
// still pinned by readers.
inline constexpr uint64_t kBudgetBytes = 64ull << 20;

// Marks objects owned by this provider; foreign pointers never match.
inline constexpr uint32_t kMagic = 0x4b564348u;  // "KVCH"

class Session;
struct Block;
struct TxRecord;

using BlockPtr = std::shared_ptr<Block>;
using TxPtr = std::shared_ptr<TxRecord>;

// Every ABI object, including the session, starts with this header and a
// KvcObject* points at it.
struct Header {
  std::atomic<uint64_t> refs{1};
  uint32_t kind = 0;  // KvcObjectKind value.
  uint32_t magic = kMagic;
  Session* session = nullptr;
};

KvcStatus OkStatus();
KvcStatus ErrorStatus(uint32_t code, std::string diagnostic);
// Wraps fn so no exception escapes a provider entry point.
template <class Fn>
KvcStatus Guard(Fn&& fn) noexcept {
  try {
    return fn();
  } catch (const std::bad_alloc&) {
    return ErrorStatus(KVC_RESOURCE_EXHAUSTED, "host provider allocation failed");
  } catch (...) {
    return ErrorStatus(KVC_INTERNAL, "host provider threw an exception");
  }
}

// One committed or reserved cache block. Committed blocks live in the
// session index; a retired generation stays alive here while read handles
// or transfers reference it. The destructor returns the payload's bytes to
// the session budget, wherever the last reference disappears.
struct Block {
  KvcBlock record{};
  std::vector<unsigned char> payload;  // Group layout: layer-major slots.
  uint64_t last_use = 0;               // LRU tick; guarded by session mutex.
  Session* session = nullptr;          // Valid while any reference exists.

  ~Block();
};

// Descriptor of one accepted transfer. Copies are synchronous here, so a
// transfer is terminal the moment it is created. The terminal code and text
// are copied into owned statuses on every poll.
struct Transfer {
  Header header{};
  std::mutex mutex;
  uint32_t state = KVC_SUCCEEDED;  // KvcTransferState value; guarded by mutex.
  uint32_t final_code = KVC_OK;    // Terminal result; guarded by mutex.
  std::string final_text;
  std::vector<BlockPtr> blocks;  // Retained cache objects.
  std::vector<Header*> regions;  // Retained imported regions.
  std::vector<Header*> deps;     // Retained dependency transfers.
};

struct Read {
  Header header{};
  std::vector<KvcBlock> manifest;  // Contiguous record view for manifest().
  std::vector<BlockPtr> blocks;    // Pinned generations, parallel to manifest.
};

struct Write {
  Header header{};
  TxPtr tx;
};

struct Region {
  Header header{};
  std::string runtime;
  std::string identifier;
  KvcMemoryInfo info{};  // Borrowed by memory_info(); strings live above.
  void* owner_context = nullptr;
  void (*owner_release)(void*) = nullptr;
};

// A write transaction, kept for the session lifetime so query_write can
// resolve the outcome of uncertain begins and commits.
struct TxRecord {
  KvcTransactionId id{};
  uint32_t state = KVC_ALLOCATED;  // KvcWriteState; guarded by session mutex.
  std::vector<KvcBlock> manifest;
  std::vector<BlockPtr> blocks;  // Parallel to manifest until abort.
  // Per-block coverage bitmap over layers * components; session mutex.
  std::vector<std::vector<bool>> covered;
  uint64_t bytes = 0;
};

class Session {
 public:
  Header header{};  // Must be first: KvcObject* is reinterpret_cast<Header*>.

  ~Session();

  // Object bookkeeping for shutdown: every non-session object counts.
  void ObjectCreated() noexcept { objects_.fetch_add(1, std::memory_order_relaxed); }
  void ObjectDestroyed() noexcept { objects_.fetch_sub(1, std::memory_order_acq_rel); }

  static Header* AsHeader(KvcObject* object) { return reinterpret_cast<Header*>(object); }
  static bool Owns(const KvcObject* object, uint32_t kind) {
    if (!object) return false;
    const auto* header = reinterpret_cast<const Header*>(object);
    return header->magic == kMagic && header->kind == kind;
  }

  std::mutex mutex;
  bool closing = false;
  bool configured = false;

  // Deep-copied configuration plus storage for everything it borrows.
  KvcConfig config{};
  std::string partition_name;
  std::vector<unsigned char> partition_payload;
  struct GroupCopy {
    KvcGroup group{};
    std::vector<uint32_t> layers;
    std::vector<KvcComponent> components;
    std::vector<std::string> component_names;  // Parallel to components.
    std::vector<uint64_t> slot_offsets;        // Component offsets in a row.
    uint64_t payload_bytes = 0;                // All layers of one block.
    std::string semantics_name;
    std::vector<unsigned char> semantics_payload;
  };
  std::vector<GroupCopy> groups;
  std::vector<KvcGroup> group_values;  // Contiguous values the borrowed config points into.

  std::unordered_map<uint64_t, BlockPtr> index;  // Key u64 prefix -> block.
  std::unordered_map<uint64_t, TxPtr> transactions;
  std::unordered_set<uint64_t> reserved_keys;
  std::atomic<uint64_t> bytes_used{0};
  std::atomic<uint64_t> lru_tick{1};
  std::atomic<uint64_t> objects_{0};

  // Observability counters (kvc.stats extension); relaxed atomics.
  std::atomic<uint64_t> stats_lookups{0};
  std::atomic<uint64_t> stats_lookup_hits{0};
  std::atomic<uint64_t> stats_lookup_hit_blocks{0};
  std::atomic<uint64_t> stats_blocks_committed{0};
  std::atomic<uint64_t> stats_blocks_retired{0};
  std::atomic<uint64_t> stats_blocks_evicted{0};
  std::atomic<uint64_t> stats_tx_begun{0};
  std::atomic<uint64_t> stats_tx_committed{0};
  std::atomic<uint64_t> stats_tx_aborted{0};
  std::atomic<uint64_t> stats_stores{0};
  std::atomic<uint64_t> stats_store_bytes{0};
  std::atomic<uint64_t> stats_loads{0};
  std::atomic<uint64_t> stats_load_bytes{0};

  const GroupCopy* FindGroup(uint32_t id) const;
  // Validates a block record against the configured groups.
  const GroupCopy* ValidateBlock(const KvcBlock& block, KvcStatus* error) const;
  static uint64_t KeyId(const KvcDigest& key);
  static uint64_t TxId(const KvcTransactionId& id);
};

// Session lifetime and negotiation (provider.cc).
KvcStatus OpenSession(const KvcDescriptor* options, uint64_t timeout, KvcObject** out);
void SessionRetain(KvcObject* object);
void SessionRelease(KvcObject* object);
KvcStatus SessionCapabilities(KvcObject* object, KvcCapabilities* out);
KvcStatus SessionConfigure(KvcObject* object, const KvcConfig* request, uint64_t timeout);
KvcStatus SessionConfiguration(KvcObject* object, const KvcConfig** out);
KvcStatus SessionShutdown(KvcObject* object, uint64_t timeout);
KvcStatus SessionQueryExtension(KvcObject* object, const KvcSchema* schema, KvcExtension* out);
KvcStatus SessionStats(KvcObject* object, KvcStatsRecord* out);

// Control plane (provider.cc).
KvcStatus Lookup(KvcObject* session, const KvcLookupRequest* request, uint64_t timeout,
                 KvcLookupResult* out);
KvcStatus BeginWrite(KvcObject* session, const KvcWriteRequest* request, uint64_t timeout,
                     KvcObject** out);
KvcStatus Commit(KvcObject* session, KvcObject* write, uint64_t timeout, KvcWriteStatus* out);
KvcStatus AbortWrite(KvcObject* session, const KvcTransactionId* id, uint64_t timeout);
KvcStatus QueryWrite(KvcObject* session, const KvcTransactionId* id, uint64_t timeout,
                     KvcWriteStatus* out);
KvcStatus RemoveBlock(KvcObject* session, const KvcDigest* key, uint64_t timeout);

// Object table (provider.cc).
void RetainObject(KvcObject* object);
void ReleaseObject(KvcObject* object);
uint32_t ObjectKind(const KvcObject* object);
KvcStatus ObjectManifest(const KvcObject* object, const KvcBlock** blocks, uint64_t* count);
KvcStatus ObjectMemoryInfo(const KvcObject* object, const KvcMemoryInfo** info);
void AbandonWrite(KvcObject* object);

// Data plane (transfer.cc).
KvcStatus ImportMemory(KvcObject* session, const KvcMemoryImport* request, KvcObject** out);
KvcStatus Load(KvcObject* session, KvcObject* read, const KvcTransferRequest* request,
               KvcObject** out);
KvcStatus Store(KvcObject* session, KvcObject* write, const KvcTransferRequest* request,
                KvcObject** out);

// Transfer table (transfer.cc).
KvcStatus PollTransfer(KvcObject* transfer, KvcTransferStatus* out);
KvcStatus WaitTransfer(KvcObject* transfer, uint64_t timeout);

// Shared binding validation for load and store. Resolves the span against
// blocks the caller pinned (load) or reserved (store) and verifies that the
// bindings exactly cover the span's (block, layer, component) slots with
// complete opaque byte ranges inside regions with the required access.
struct BindingPlan {
  struct Entry {
    size_t block_position;  // Index into blocks.
    uint64_t payload_offset;
    unsigned char* host_address;
    uint64_t byte_count;
  };
  const Session::GroupCopy* group = nullptr;
  std::vector<BlockPtr> blocks;  // Resolved blocks, span order.
  std::vector<Entry> entries;
};
KvcStatus ResolveBindings(Session& session, const KvcSpan& span, const KvcBinding* bindings,
                          uint64_t binding_count, const std::vector<BlockPtr>& owned,
                          uint32_t required_access, bool writing,
                          std::vector<std::vector<bool>>* coverage, BindingPlan* out);

}  // namespace kvc_host

#endif  // KVCACHE_HOST_INTERNAL_H_
