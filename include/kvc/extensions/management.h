// Optional "kvc.management" extension, version 1.0.
//
// Administrative inspection and bulk retirement of committed state — the
// surface both engines expose out-of-band (vLLM reset_prefix_cache, needed
// after weight updates; SGLang /flush_cache and its hicache storage list
// endpoints). remove() retires one digest at a time; these operations are
// session-scoped because a session's namespace and model identity are
// fixed at configuration.
//
// Queried through the provider's query_extension entry point:
//
//   auto ext = provider.QueryExtension({{"kvc.management", 15}, 1, 0});
//   if (ext) {
//     auto* api = static_cast<const KvcManagementApi*>(ext->Table());
//     uint64_t total = 0;
//     api->enumerate(ext->NativeHandle(), nullptr, 0, &total);
//   }

#ifndef KVC_EXTENSIONS_MANAGEMENT_H_
#define KVC_EXTENSIONS_MANAGEMENT_H_

#include <stdint.h>

#include "kvc/abi/basics.h"
#include "kvc/abi/provider.h"

#ifdef __cplusplus
extern "C" {
#endif

#define KVC_MANAGEMENT_SCHEMA_NAME "kvc.management"
#define KVC_MANAGEMENT_SCHEMA_NAME_LEN 15
#define KVC_MANAGEMENT_SCHEMA_MAJOR 1u
#define KVC_MANAGEMENT_SCHEMA_MINOR 0u

/// \brief One committed block's identity, as reported by enumerate().
typedef struct KvcStoredBlock {
  uint32_t struct_size;
  KvcBlock record;
} KvcStoredBlock;

/// \brief Versioned table of the kvc.management extension. The owner
/// object is the session being inspected or cleared.
typedef struct KvcManagementApi {
  uint32_t struct_size;
  uint32_t abi_major;
  uint32_t abi_minor;
  /// \brief Reports committed blocks' records.
  ///
  /// Fills at most `max` entries in `out` (any order; pass null to size
  /// only) and always writes the committed count to `out_total`. Blocks
  /// pinned by read handles are included: they are committed state.
  /// Thread-safe against concurrent publication; the result is a snapshot.
  KvcStatus(KVC_CALL* enumerate)(KvcObject* owner, KvcStoredBlock* out, uint64_t max,
                                 uint64_t* out_total);
  /// \brief Retires every committed block of the session (bulk remove).
  ///
  /// Idempotent retirement: existing read handles keep their pinned
  /// generations; blocks become undiscoverable immediately. `out_removed`
  /// (nullable) receives how many entries retired.
  KvcStatus(KVC_CALL* clear)(KvcObject* owner, uint64_t* out_removed);
} KvcManagementApi;

#ifdef __cplusplus
}
#endif

#endif  // KVC_EXTENSIONS_MANAGEMENT_H_
