/* Minimal provider-author scaffold in C. Discovery and ownership are real;
 * cache operations explicitly return Unsupported until a backend is supplied.
 * See plugins/kvcache/host for a functional reference implementation. */
#include "kvc/abi/provider.h"

#include <stdatomic.h>
#include <stdlib.h>

struct KvcObject {
  atomic_uint refs;
};
static KvcStatus Result(uint32_t code) { return (KvcStatus){.code = code}; }
static KvcStatus KVC_CALL OpenSession(const KvcDescriptor* options, uint64_t timeout,
                                      KvcObject** out) {
  (void)options;
  (void)timeout;
  if (!out) return Result(KVC_INVALID_ARGUMENT);
  *out = calloc(1, sizeof(KvcObject));
  if (!*out) return Result(KVC_RESOURCE_EXHAUSTED);
  atomic_init(&(*out)->refs, 1);
  return Result(KVC_OK);
}
static void KVC_CALL Retain(KvcObject* object) { atomic_fetch_add(&object->refs, 1); }
static void KVC_CALL Release(KvcObject* object) {
  if (atomic_fetch_sub(&object->refs, 1) == 1) free(object);
}
static uint32_t KVC_CALL Kind(const KvcObject* object) { return object ? KVC_SESSION : 0; }
static KvcStatus KVC_CALL Manifest(const KvcObject* object, const KvcBlock** out, uint64_t* n) {
  (void)object;
  *out = NULL;
  *n = 0;
  return Result(KVC_STALE_HANDLE);
}
static KvcStatus KVC_CALL MemoryInfo(const KvcObject* object, const KvcMemoryInfo** out) {
  (void)object;
  *out = NULL;
  return Result(KVC_STALE_HANDLE);
}
static void KVC_CALL Abandon(KvcObject* object) { (void)object; }
static KvcStatus KVC_CALL Capabilities(KvcObject* object, KvcCapabilities* out) {
  (void)object;
  if (!out || out->struct_size < sizeof(*out)) return Result(KVC_INVALID_ARGUMENT);
  *out = (KvcCapabilities){.struct_size = sizeof(*out)};
  return Result(KVC_OK);
}
static KvcStatus KVC_CALL Configure(KvcObject* s, const KvcConfig* r, uint64_t t) {
  (void)s;
  (void)r;
  (void)t;
  return Result(KVC_UNSUPPORTED);
}
static KvcStatus KVC_CALL Configuration(KvcObject* s, const KvcConfig** out) {
  (void)s;
  *out = NULL;
  return Result(KVC_INVALID_ARGUMENT);
}
static KvcStatus KVC_CALL ShutdownSession(KvcObject* s, uint64_t t) {
  (void)s;
  (void)t;
  return Result(KVC_OK);
}
static KvcStatus KVC_CALL Lookup(KvcObject* s, const KvcLookupRequest* r, uint64_t t,
                                 KvcLookupResult* out) {
  (void)s;
  (void)r;
  (void)t;
  out->read = NULL;
  return Result(KVC_UNSUPPORTED);
}
static KvcStatus KVC_CALL Begin(KvcObject* s, const KvcWriteRequest* r, uint64_t t,
                                KvcObject** out) {
  (void)s;
  (void)r;
  (void)t;
  *out = NULL;
  return Result(KVC_UNSUPPORTED);
}
static KvcStatus KVC_CALL Commit(KvcObject* s, KvcObject* w, uint64_t t, KvcWriteStatus* out) {
  (void)s;
  (void)w;
  (void)t;
  (void)out;
  return Result(KVC_UNSUPPORTED);
}
static KvcStatus KVC_CALL AbortWrite(KvcObject* s, const KvcTransactionId* id, uint64_t t) {
  (void)s;
  (void)id;
  (void)t;
  return Result(KVC_UNSUPPORTED);
}
static KvcStatus KVC_CALL Query(KvcObject* s, const KvcTransactionId* id, uint64_t t,
                                KvcWriteStatus* out) {
  (void)s;
  (void)id;
  (void)t;
  (void)out;
  return Result(KVC_UNSUPPORTED);
}
static KvcStatus KVC_CALL RemoveBlock(KvcObject* s, const KvcDigest* key, uint64_t t) {
  (void)s;
  (void)key;
  (void)t;
  return Result(KVC_UNSUPPORTED);
}
static KvcStatus KVC_CALL ImportMemory(KvcObject* s, const KvcMemoryImport* r,
                                       KvcObject** out) {
  (void)s;
  (void)r;
  *out = NULL;
  return Result(KVC_UNSUPPORTED);
}
static KvcStatus KVC_CALL Transfer(KvcObject* s, KvcObject* h, const KvcTransferRequest* r,
                                   KvcObject** out) {
  (void)s;
  (void)h;
  (void)r;
  *out = NULL;
  return Result(KVC_UNSUPPORTED);
}
static KvcStatus KVC_CALL Poll(KvcObject* h, KvcTransferStatus* out) {
  (void)h;
  (void)out;
  return Result(KVC_STALE_HANDLE);
}
static KvcStatus KVC_CALL WaitTransfer(KvcObject* h, uint64_t t) {
  (void)h;
  (void)t;
  return Result(KVC_STALE_HANDLE);
}
#define HEADER(T) sizeof(T), KVC_ABI_MAJOR, KVC_ABI_MINOR
static const KvcObjectApi objects = {HEADER(KvcObjectApi), Retain, Release, Kind, Manifest,
                                     MemoryInfo,           Abandon};
static const KvcProviderApi providers = {HEADER(KvcProviderApi), Capabilities,    Configure,
                                         Configuration,          ShutdownSession, NULL};
static const KvcControl controls = {HEADER(KvcControl), Lookup, Begin,      Commit,
                                    AbortWrite,         Query,  RemoveBlock};
static const KvcData data = {HEADER(KvcData), ImportMemory, Transfer, Transfer, NULL};
static const KvcTransferApi transfers = {HEADER(KvcTransferApi), Poll, WaitTransfer, NULL};
static const KvcPlugin plugin = {HEADER(KvcPlugin), {"example.scaffold", 16},
                                 OpenSession,       &providers,
                                 &controls,         &data,
                                 &objects,          &transfers};

KVC_EXPORT KvcStatus KVC_CALL KvcGetProvider(uint32_t major, uint32_t minor,
                                             const KvcPlugin** out) {
  (void)minor;
  if (!out) return Result(KVC_INVALID_ARGUMENT);
  *out = NULL;
  if (major != KVC_ABI_MAJOR) return Result(KVC_UNSUPPORTED);
  *out = &plugin;
  return Result(KVC_OK);
}
