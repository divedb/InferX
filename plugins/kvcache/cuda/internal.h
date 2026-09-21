// CUDA tiered provider: GPU-resident KV storage with a CPU offload tier.
//
// Committed blocks live in device memory slots while hot. Under GPU budget
// pressure the least-recently-used committed blocks are written back to a
// host-memory tier; under host budget pressure the least-recently-used host
// blocks are retired from the index. Loads promote host-resident blocks
// back into device slots when room can be made. Device copies are enqueued
// on one internal stream and complete through CUDA events; transfers that
// touch host memory synchronously are completed before submission returns.
//
// The provider implements the same fixed opaque-component profile as the
// host reference provider (see docs/extensions/kvcache.md).

#ifndef KVCACHE_CUDA_INTERNAL_H_
#define KVCACHE_CUDA_INTERNAL_H_

#include <cuda_runtime.h>

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

namespace kvc_cuda {

inline constexpr uint32_t kMagic = 0x4b564344u;  // "KVCD"
inline constexpr uint64_t kDefaultHostBudget = 1ull << 30;
inline constexpr uint64_t kMaxAutoGpuBudget = 2ull << 30;

KvcStatus OkStatus();
KvcStatus ErrorStatus(uint32_t code, std::string diagnostic);
template <class Fn>
KvcStatus Guard(Fn&& fn) noexcept {
  try {
    return fn();
  } catch (const std::bad_alloc&) {
    return ErrorStatus(KVC_RESOURCE_EXHAUSTED, "cuda provider allocation failed");
  } catch (...) {
    return ErrorStatus(KVC_INTERNAL, "cuda provider threw an exception");
  }
}

class Session;
struct Block;
struct TxRecord;

using BlockPtr = std::shared_ptr<Block>;
using TxPtr = std::shared_ptr<TxRecord>;

struct Header {
  std::atomic<uint64_t> refs{1};
  uint32_t kind = 0;
  uint32_t magic = kMagic;
  Session* session = nullptr;
};

// Free-list slot allocator for one block-payload size class. Slots are
// cudaMalloc'd on demand and returned here instead of freed, so repeated
// publication and eviction do not fragment the device heap.
struct SlotPool {
  explicit SlotPool(uint64_t bytes) : slot_bytes(bytes) {}
  ~SlotPool();

  const uint64_t slot_bytes;
  int device = 0;
  std::mutex mutex;
  std::vector<void*> free_slots;
  size_t live = 0;                       // Slots owned by blocks; guarded by mutex.
  size_t total = 0;                      // Slots ever allocated; guarded by mutex.
  std::atomic<uint64_t> allocations{0};  // Fresh cudaMalloc'd slots.
  std::atomic<uint64_t> reuses{0};       // Slots served from the free list.

  void* Take();           // Null on allocation failure.
  void Give(void* slot);  // Returns a slot to the free list.
  uint64_t LiveBytes();
};

class Session;

struct Block {
  KvcBlock record{};
  uint64_t payload_bytes = 0;
  // Placement: exactly one of gpu slot or host buffer holds the bytes.
  void* gpu = nullptr;              // Device slot base; null if offloaded.
  std::shared_ptr<SlotPool> pool;   // Owning pool while gpu is set.
  std::vector<unsigned char> host;  // Offload tier storage.
  uint64_t last_use = 0;            // LRU tick; session mutex.
  uint32_t pending = 0;             // Live transfers touching storage.
  bool tx_private = true;           // Until commit publishes it.
  Session* session = nullptr;       // Valid while any reference exists.

  // Releases storage and its budget accounting. The caller holds the
  // session mutex; idempotent.
  void ReclaimStorage();
  // Blocks die only after every transfer referencing them completed, so
  // pending is zero here and reclaiming under the session lock is safe.
  ~Block();
};

struct Transfer;

struct Read {
  Header header{};
  std::vector<KvcBlock> manifest;
  std::vector<BlockPtr> blocks;
};

struct Write {
  Header header{};
  TxPtr tx;
};

struct Region {
  Header header{};
  std::string runtime;
  std::string identifier;
  KvcMemoryInfo info{};
  void* owner_context = nullptr;
  void (*owner_release)(void*) = nullptr;
};

struct TxRecord {
  KvcTransactionId id{};
  uint32_t state = KVC_ALLOCATED;  // Session mutex.
  std::vector<KvcBlock> manifest;
  std::vector<BlockPtr> blocks;
  std::vector<std::vector<bool>> covered;  // Per block, layer*component bits.
  std::vector<Header*> transfers;          // Store transfers; session mutex.
  uint64_t bytes = 0;
  uint64_t reserved_remainder = 0;  // Reserved bytes without a slot yet.
};

// One accepted transfer. state is atomic: PENDING until exactly one
// observer (poll, wait, or release) completes it. final_code/final_text
// are written before the state store and read only after it, under mutex.
struct Transfer {
  Header header{};
  std::mutex mutex;  // Guards final_text and the retained vectors.
  std::atomic<uint32_t> state{KVC_PENDING};
  uint32_t final_code = KVC_OK;
  std::string final_text;
  cudaEvent_t event = nullptr;  // Null when no async work was enqueued.
  bool poisoned_tx = false;     // Store transfers poison on failure.
  TxPtr tx;                     // Set for store transfers.
  std::vector<BlockPtr> blocks;
  std::vector<Header*> regions;
  std::vector<Header*> deps;
};

class Session {
 public:
  Header header{};  // Must be first.

  ~Session();

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

  int device_ordinal = 0;
  uint64_t gpu_budget = 0;
  uint64_t host_budget = kDefaultHostBudget;

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
    std::string semantics_name;
    std::vector<unsigned char> semantics_payload;
  };
  std::vector<GroupCopy> groups;
  std::vector<KvcGroup> group_values;  // Contiguous values the borrowed config points into.

  std::unordered_map<uint64_t, BlockPtr> index;
  std::unordered_map<uint64_t, TxPtr> transactions;
  std::unordered_set<uint64_t> reserved_keys;
  std::unordered_map<uint64_t, std::shared_ptr<SlotPool>> pools;  // By slot bytes.
  uint64_t gpu_reserved = 0;  // Begin-write reservations not yet allocated.
  std::atomic<uint64_t> host_bytes{0};
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
  std::atomic<uint64_t> stats_offloads{0};
  std::atomic<uint64_t> stats_offload_bytes{0};
  std::atomic<uint64_t> stats_promotions{0};
  std::atomic<uint64_t> stats_promotion_bytes{0};

  cudaStream_t stream = nullptr;

  const GroupCopy* FindGroup(uint32_t id) const;
  const GroupCopy* ValidateBlock(const KvcBlock& block, KvcStatus* error) const;
  static uint64_t KeyId(const KvcDigest& key);
  static uint64_t TxId(const KvcTransactionId& id);

  // Tiering, all under the session mutex (provider.cc).
  uint64_t GpuUsedLocked();
  KvcStatus MakeGpuRoomLocked(uint64_t bytes);
  KvcStatus MakeHostRoomLocked(uint64_t bytes);
  KvcStatus OffloadLocked(const BlockPtr& block);
  bool TryPromoteLocked(const BlockPtr& block);
  void ReclaimLocked(const BlockPtr& block);  // Idempotent storage release.

  // Shared plumbing (provider.cc).
  SlotPool* PoolForLocked(uint64_t slot_bytes);
  // Completes a transfer exactly once and applies block bookkeeping.
  void FinishTransferLocked(Transfer* transfer);
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
KvcStatus PollTransfer(KvcObject* transfer, KvcTransferStatus* out);
KvcStatus WaitTransfer(KvcObject* transfer, uint64_t timeout);
// Completes a pending transfer if its event allows: queries the CUDA event,
// flips the state exactly once, and applies block bookkeeping by taking the
// session lock. Returns the current KvcTransferState value.
uint32_t ObserveTransfer(Transfer* transfer);
// Same, for callers already holding the session mutex.
uint32_t ObserveTransferLocked(Transfer* transfer);
// Waits for a pending transfer's event, then observes it. Used on the
// release path so reclaimed resources are never still being accessed.
void DrainTransfer(Transfer* transfer);

}  // namespace kvc_cuda

#endif  // KVCACHE_CUDA_INTERNAL_H_
