#ifndef KVC_ABI_BASICS_H_
#define KVC_ABI_BASICS_H_

#include <stdint.h>

#if defined(_WIN32)
#define KVC_CALL __cdecl
#define KVC_EXPORT __declspec(dllexport)
#else
#define KVC_CALL
#define KVC_EXPORT __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

/// ABI version implemented by these headers. Providers reject a mismatched
/// major at the entry point; minor versions append fields only.
#define KVC_ABI_MAJOR 1u
#define KVC_ABI_MINOR 0u

/// Timeout value meaning "wait indefinitely". Timeouts are nanoseconds;
/// zero means do not wait.
#define KVC_TIMEOUT_INFINITE UINT64_MAX

// Enumerators are named like macros and stored in fixed-width record fields
// (uint32_t/uint64_t); the enum tags below name the value sets, not the
// storage. Record fields keep their integer types so adding enumerators
// never changes struct layouts.

/// \brief Result codes returned by every ABI operation. Zero is success;
///        every other value names a specific failure with defined recovery.
enum KvcCode {
  KVC_OK = 0,              ///< Success.
  KVC_NOT_FOUND,           ///< Requested state is not present.
  KVC_INVALID_ARGUMENT,    ///< Malformed request; no state was touched.
  KVC_INCOMPATIBLE_MODEL,  ///< Cached state belongs to different content.
  KVC_INVALID_RANGE,       ///< Indices or extents outside their bounds.
  KVC_UNSUPPORTED,         ///< Operation or profile is not implemented.
  KVC_RESOURCE_EXHAUSTED,  ///< Capacity or budget is exhausted.
  KVC_TRANSPORT_ERROR,     ///< Data movement failed midway.
  KVC_UNAVAILABLE,         ///< Backing service is unreachable.
  KVC_DATA_LOSS,           ///< Previously acknowledged state was lost.
  KVC_STALE_HANDLE,        ///< Handle is invalid, freed, or foreign.
  KVC_ALREADY_EXISTS,      ///< State with this identity is already present.
  KVC_CONFLICT,            ///< Concurrent operation owns the resource.
  KVC_ALREADY_COMMITTED,   ///< Transaction was committed before this call.
  KVC_ABORTED,             ///< Transaction was aborted before this call.
  KVC_BUSY,                ///< Outstanding work blocks the operation.
  KVC_INCOMPLETE,          ///< Write is missing required stored components.
  KVC_DEADLINE_EXCEEDED,   ///< Timeout elapsed before completion.
  KVC_OUTCOME_UNKNOWN,     ///< Operation may or may not have taken effect.
  KVC_CANCELLED,           ///< Cancellation succeeded before completion.
  KVC_SHUTTING_DOWN,       ///< Session is closing and admits no new work.
  KVC_INTERNAL             ///< Provider-internal failure. Report as a bug.
};

/// \brief Kind of an opaque object, reported by the object table's kind()
///        and stored in every handle's identity.
enum KvcObjectKind {
  KVC_SESSION = 1,  ///< Configured provider session.
  KVC_READ,         ///< Pinned set of committed blocks (ReadHandle).
  KVC_WRITE,        ///< Exclusive unpublished transaction (WriteHandle).
  KVC_MEMORY,       ///< Imported memory region (MemoryRegion).
  KVC_TRANSFER      ///< Accepted transfer awaiting completion.
};

/// \brief Opaque handle base. Every session, read, write, region, and
///        transfer is a KvcObject*; only the provider may dereference it.
typedef struct KvcObject KvcObject;

/// \brief Borrowed read-only byte span. Not NUL-terminated; valid for the
///        duration of the call unless a documented owner retains it.
typedef struct KvcBytes {
  const unsigned char* data;
  uint64_t size;
} KvcBytes;

/// \brief Borrowed read-only UTF-8 string span. Not NUL-terminated.
typedef struct KvcString {
  const char* data;
  uint64_t size;
} KvcString;

/// \brief Fixed 32-byte identity value, stored inline rather than borrowed.
///
/// Used for cache namespaces, model identities, block keys, and prefix
/// dependencies. The containing field determines which identity it names;
/// the digest is not the cached K/V payload or a memory address.
///
/// Callers produce identities using an agreed canonical encoding and hash
/// scheme. A block key must account for its model, semantics, partition,
/// and input dependencies, not merely its byte count or tensor shape.
/// Never hash raw struct memory: padding and pointers are not portable.
/// This record holds the result; it does not compute or validate a digest.
typedef struct KvcDigest {
  unsigned char bytes[32];
} KvcDigest;

/// \brief Name and version of a semantic schema. Unknown names must be
///        negotiated explicitly and never silently reinterpreted.
///
/// EXAMPLE:
/// \code{.c}
/// KvcSchema schema = {{"kvc.bytes", sizeof("kvc.bytes") - 1}, 1, 0};
/// KvcDescriptor meaning = {schema, {0, 0}};
/// \endcode
///
/// This selects the "kvc.bytes" component schema, version 1.0, whose
/// canonical parameters are empty. Assign meaning to KvcComponent::meaning
/// with layout KVC_OPAQUE_BYTES and the desired bytes_per_block. The schema
/// describes how to interpret the component; the cached bytes are supplied
/// separately through memory-region bindings. The name length excludes the
/// trailing NUL, and the string literal keeps the borrowed name alive.
typedef struct KvcSchema {
  KvcString name;
  uint32_t major;
  uint32_t minor;
} KvcSchema;

/// \brief Semantic identity: a schema plus its canonical parameters. Names
/// what component bytes mean (e.g. "kvc.bytes") and what partition they
/// belong to, independent of storage geometry.
typedef struct KvcDescriptor {
  KvcSchema schema;
  KvcBytes canonical_payload;
} KvcDescriptor;

/// \brief Operation result. code is a KvcCode value; diagnostic is owned by
/// the provider and must be freed exactly once through release_diagnostic.
typedef struct KvcStatus {
  uint32_t code;
  KvcString diagnostic;
  void* diagnostic_owner;
  void(KVC_CALL* release_diagnostic)(void*);
} KvcStatus;

/// \brief Reference-counted allocation owner handed to the provider. The
/// callbacks must not throw or call back into the provider; release drops
/// the provider's last reference to the allocation.
typedef struct KvcOwner {
  void* context;
  void(KVC_CALL* retain)(void*);
  void(KVC_CALL* release)(void*);
} KvcOwner;

// Fixed records are immutable values. Extensible call records and tables
// carry struct_size, initialized to sizeof(record) by the caller. Arrays
// and strings are borrowed during a call unless a documented parent object
// retains their lifetime. Never hash struct memory.

#ifdef __cplusplus
}
#endif
#endif  // KVC_ABI_BASICS_H_
