#ifndef KVC_ABI_PROVIDER_H_
#define KVC_ABI_PROVIDER_H_

#include "kvc/abi/types.h"

#ifdef __cplusplus
extern "C" {
#endif

/// The dynamic symbol the loader resolves with dlsym/GetProcAddress.
#define KVC_PROVIDER_ENTRY_SYMBOL "KvcGetProvider"

/// Common prefix of every function table: the caller-visible extent of the
/// struct and the ABI version it was built against. Tables grow by
/// appending fields; loaders reject a mismatched major and ignore unknown
/// trailing fields.
#define KVC_TABLE_HEADER \
  uint32_t struct_size;  \
  uint32_t abi_major;    \
  uint32_t abi_minor

/// \brief Lifetime, identity, and metadata of opaque objects. Every
/// non-null object reference obtained from another table is eventually
/// returned through release here.
typedef struct KvcObjectApi {
  KVC_TABLE_HEADER;
  /// Acquires a reference. Null is permitted and ignored.
  void(KVC_CALL* retain)(KvcObject*);
  /// Drops a reference, destroying the object at zero. A write destroyed
  /// while uncommitted is aborted.
  void(KVC_CALL* release)(KvcObject*);
  /// Reports the KvcObjectKind of a live object, 0 for foreign pointers.
  uint32_t(KVC_CALL* kind)(const KvcObject*);
  /// Returns the block manifest of a read or write handle; borrowed until
  /// the handle's release.
  KvcStatus(KVC_CALL* manifest)(const KvcObject*, const KvcBlock**, uint64_t*);
  /// Returns the stable memory description of a region; borrowed until the
  /// region's release.
  KvcStatus(KVC_CALL* memory_info)(const KvcObject*, const KvcMemoryInfo**);
  // Nonblocking abandonment, separate from reference counting: drops an
  // uncommitted write's reservation without waiting for pending stores.
  void(KVC_CALL* abandon_write)(KvcObject*);
} KvcObjectApi;

/// \brief Control plane: discovery, transactions, publication, and
/// retirement of cached state. All entries are required.
typedef struct KvcControl {
  KVC_TABLE_HEADER;
  /// Evaluates every candidate; returns the longest fully available one
  /// as a pinned read handle, or a normal miss (NULL read).
  KvcStatus(KVC_CALL* lookup)(KvcObject*, const KvcLookupRequest*, uint64_t, KvcLookupResult*);
  /// Reserves a manifest privately under a fresh transaction id; nothing
  /// is visible to lookup until commit.
  KvcStatus(KVC_CALL* begin_write)(KvcObject*, const KvcWriteRequest*, uint64_t, KvcObject**);
  /// Publishes every block of a transaction atomically, once all selected
  /// components are stored; repeat calls return the same receipt.
  KvcStatus(KVC_CALL* commit)(KvcObject*, KvcObject*, uint64_t, KvcWriteStatus*);
  /// Rolls back an unfinished transaction by id.
  KvcStatus(KVC_CALL* abort)(KvcObject*, const KvcTransactionId*, uint64_t);
  /// Reports a transaction's state and receipt by id.
  KvcStatus(KVC_CALL* query_write)(KvcObject*, const KvcTransactionId*, uint64_t,
                                   KvcWriteStatus*);
  /// Retires a block key: removes discoverability; existing pins stay
  /// valid until released.
  KvcStatus(KVC_CALL* remove)(KvcObject*, const KvcDigest*, uint64_t);
} KvcControl;

/// \brief Data plane: importing memory and moving cached state. load and
/// store return a transfer handle and retain their dependencies until
/// quiescence.
typedef struct KvcData {
  KVC_TABLE_HEADER;
  /// Adopts an allocation described by the caller, kept alive through
  /// allocation_owner.
  KvcStatus(KVC_CALL* import_memory)(KvcObject*, const KvcMemoryImport*, KvcObject**);
  /// Copies cached state into caller memory bound to a read handle.
  KvcStatus(KVC_CALL* load)(KvcObject*, KvcObject*, const KvcTransferRequest*, KvcObject**);
  /// Copies caller memory into a write transaction's reserved blocks.
  KvcStatus(KVC_CALL* store)(KvcObject*, KvcObject*, const KvcTransferRequest*, KvcObject**);
  // Optional entries may be NULL; consumers return Unsupported.
  /// Maps immutable cached state for direct reads. Requires the Mapping
  /// feature.
  KvcStatus(KVC_CALL* map_read)(KvcObject*, KvcObject*, const KvcSpan*, KvcMappedRead*);
} KvcData;

/// \brief Completion interface of an accepted transfer.
typedef struct KvcTransferApi {
  KVC_TABLE_HEADER;
  /// Reports current state without waiting.
  KvcStatus(KVC_CALL* poll)(KvcObject*, KvcTransferStatus*);
  /// Waits for a terminal state, up to the timeout.
  KvcStatus(KVC_CALL* wait)(KvcObject*, uint64_t);
  /// Requests cancellation. Optional: may be NULL. Success means terminal
  /// Cancelled; failure does not cancel.
  KvcStatus(KVC_CALL* cancel)(KvcObject*);
} KvcTransferApi;

/// \brief Session-level negotiation and lifecycle. All entries except
/// query_extension are required.
typedef struct KvcProviderApi {
  KVC_TABLE_HEADER;
  /// Reports features, memory types, layouts, and supported schemas.
  KvcStatus(KVC_CALL* capabilities)(KvcObject*, KvcCapabilities*);
  /// Validates and installs the one configuration; deep-copies it.
  KvcStatus(KVC_CALL* configure)(KvcObject*, const KvcConfig*, uint64_t);
  /// Returns the installed configuration; borrowed until session release.
  KvcStatus(KVC_CALL* configuration)(KvcObject*, const KvcConfig**);
  /// Drains or fences: returns Busy while owners or background work
  /// remain, and success only when no provider code can still run.
  KvcStatus(KVC_CALL* shutdown)(KvcObject*, uint64_t);
  // Optional.
  /// Looks up a vendor extension table by schema name and version.
  KvcStatus(KVC_CALL* query_extension)(KvcObject*, const KvcSchema*, KvcExtension*);
} KvcProviderApi;

/// \brief The plugin record returned by the entry symbol: identity,
/// factory, and the operation tables. All tables are immutable and live
/// until library unload.
typedef struct KvcPlugin {
  KVC_TABLE_HEADER;
  /// Stable provider identity (e.g. "kvc.host"); UTF-8, not
  /// NUL-terminated.
  KvcString provider_id;
  /// Creates one session object; options may carry provider-specific
  /// open parameters.
  KvcStatus(KVC_CALL* open)(const KvcDescriptor*, uint64_t, KvcObject**);
  const KvcProviderApi* provider;
  const KvcControl* control;
  const KvcData* data;
  const KvcObjectApi* object;
  const KvcTransferApi* transfer;
} KvcPlugin;

/// \brief Pointer type of the entry symbol, for dlsym-style loaders.
// Entry is pure discovery. Returned tables live until library unload. Major
// mismatch returns Unsupported and NULL. New minor versions append fields.
// No exceptions may escape any entry point, callback, or release function.
typedef KvcStatus(KVC_CALL* KvcGetProviderFn)(uint32_t, uint32_t, const KvcPlugin**);

/// \brief The single exported symbol of a provider library
/// (KVC_PROVIDER_ENTRY_SYMBOL). Receives the consumer's requested ABI
/// version and returns the plugin record, or Unsupported with *out = NULL
/// on a major mismatch. Pure discovery: no caches, sessions, or threads
/// are touched here.
KvcStatus KVC_CALL KvcGetProvider(uint32_t requested_major, uint32_t requested_minor,
                                  const KvcPlugin** out);

#undef KVC_TABLE_HEADER
#ifdef __cplusplus
}
#endif
#endif  // KVC_ABI_PROVIDER_H_
