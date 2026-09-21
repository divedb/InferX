// Optional "kvc.stats" extension, version 1.0.
//
// Providers that advertise this extension report lifetime counters and
// current tier occupancy for observability. The extension is queried
// through the provider's query_extension entry point:
//
//   auto ext = provider.QueryExtension({{"kvc.stats", 9}, 1, 0});
//   if (ext) {
//     auto* api = static_cast<const KvcStatsApi*>(ext->Table());
//     KvcStatsRecord record{};
//     record.struct_size = sizeof(record);
//     api->get(ext->NativeHandle(), &record);
//   }
//
// Counters are monotonically increasing for the session lifetime except the
// *_in_use and *_budget fields, which are current snapshots. Providers
// without a host or device tier report zeros for the inapplicable fields.
// Unknown trailing fields (from newer minor versions) are zero-initialized
// by the provider.

#ifndef KVC_EXTENSIONS_STATS_H_
#define KVC_EXTENSIONS_STATS_H_

#include "kvc/abi/provider.h"

#ifdef __cplusplus
extern "C" {
#endif

#define KVC_STATS_SCHEMA_NAME "kvc.stats"
#define KVC_STATS_SCHEMA_NAME_LEN 9
#define KVC_STATS_SCHEMA_MAJOR 1u
#define KVC_STATS_SCHEMA_MINOR 0u

/// \brief Lifetime counters and tier occupancy of a provider session.
typedef struct KvcStatsRecord {
  uint32_t struct_size;
  uint64_t lookups;            ///< Lookup calls evaluated.
  uint64_t lookup_hits;        ///< Lookups that pinned a candidate.
  uint64_t lookup_hit_blocks;  ///< Blocks covered by hits.
  uint64_t blocks_committed;   ///< Blocks published by commit.
  uint64_t blocks_retired;     ///< Blocks removed by remove().
  uint64_t blocks_evicted;     ///< Blocks dropped for tier capacity.
  uint64_t transactions_begun;
  uint64_t transactions_committed;
  uint64_t transactions_aborted;
  uint64_t stores;           ///< Accepted store transfers.
  uint64_t store_bytes;      ///< Payload bytes of accepted stores.
  uint64_t loads;            ///< Accepted load transfers.
  uint64_t load_bytes;       ///< Payload bytes of accepted loads.
  uint64_t gpu_allocations;  ///< Fresh device-slot allocations.
  uint64_t gpu_reuses;       ///< Slots served from the free list.
  uint64_t offloads;         ///< GPU-to-host tier moves.
  uint64_t offload_bytes;
  uint64_t promotions;  ///< Host-to-GPU restores on load.
  uint64_t promotion_bytes;
  uint64_t gpu_bytes_in_use;   ///< Device tier occupancy (snapshot).
  uint64_t gpu_budget;         ///< Device tier capacity.
  uint64_t host_bytes_in_use;  ///< Host tier occupancy (snapshot).
  uint64_t host_budget;        ///< Host tier capacity.
} KvcStatsRecord;

/// \brief Versioned table of the kvc.stats extension. The owner object is
/// the session the statistics describe.
typedef struct KvcStatsApi {
  uint32_t struct_size;
  uint32_t abi_major;
  uint32_t abi_minor;
  /// Fills `out` with the session's current statistics. Thread-safe.
  KvcStatus(KVC_CALL* get)(KvcObject* owner, KvcStatsRecord* out);
} KvcStatsApi;

#ifdef __cplusplus
}
#endif

#endif  // KVC_EXTENSIONS_STATS_H_
