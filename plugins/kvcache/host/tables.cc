// ABI function tables and the loader entry point of the host provider.

#include "internal.h"

namespace kvc_host {
namespace {

KvcStatus KVC_CALL Open(const KvcDescriptor* options, uint64_t timeout, KvcObject** out) {
  return OpenSession(options, timeout, out);
}

KvcStatus KVC_CALL Capabilities(KvcObject* object, KvcCapabilities* out) {
  return SessionCapabilities(object, out);
}
KvcStatus KVC_CALL Configure(KvcObject* object, const KvcConfig* request, uint64_t timeout) {
  return SessionConfigure(object, request, timeout);
}
KvcStatus KVC_CALL Configuration(KvcObject* object, const KvcConfig** out) {
  return SessionConfiguration(object, out);
}
KvcStatus KVC_CALL Shutdown(KvcObject* object, uint64_t timeout) {
  return SessionShutdown(object, timeout);
}
KvcStatus KVC_CALL QueryExtension(KvcObject* object, const KvcSchema* schema,
                                  KvcExtension* out) {
  return SessionQueryExtension(object, schema, out);
}

KvcStatus KVC_CALL Lookup(KvcObject* object, const KvcLookupRequest* request, uint64_t timeout,
                          KvcLookupResult* out) {
  return kvc_host::Lookup(object, request, timeout, out);
}
KvcStatus KVC_CALL BeginWrite(KvcObject* object, const KvcWriteRequest* request,
                              uint64_t timeout, KvcObject** out) {
  return kvc_host::BeginWrite(object, request, timeout, out);
}
KvcStatus KVC_CALL Commit(KvcObject* object, KvcObject* write, uint64_t timeout,
                          KvcWriteStatus* out) {
  return kvc_host::Commit(object, write, timeout, out);
}
KvcStatus KVC_CALL Abort(KvcObject* object, const KvcTransactionId* id, uint64_t timeout) {
  return AbortWrite(object, id, timeout);
}
KvcStatus KVC_CALL QueryWrite(KvcObject* object, const KvcTransactionId* id, uint64_t timeout,
                              KvcWriteStatus* out) {
  return kvc_host::QueryWrite(object, id, timeout, out);
}
KvcStatus KVC_CALL Remove(KvcObject* object, const KvcDigest* key, uint64_t timeout) {
  return RemoveBlock(object, key, timeout);
}

KvcStatus KVC_CALL Import(KvcObject* object, const KvcMemoryImport* request, KvcObject** out) {
  return ImportMemory(object, request, out);
}
KvcStatus KVC_CALL Load(KvcObject* object, KvcObject* read, const KvcTransferRequest* request,
                        KvcObject** out) {
  return kvc_host::Load(object, read, request, out);
}
KvcStatus KVC_CALL Store(KvcObject* object, KvcObject* write, const KvcTransferRequest* request,
                         KvcObject** out) {
  return kvc_host::Store(object, write, request, out);
}

void KVC_CALL Retain(KvcObject* object) { RetainObject(object); }
void KVC_CALL Release(KvcObject* object) { ReleaseObject(object); }
uint32_t KVC_CALL Kind(const KvcObject* object) { return ObjectKind(object); }
KvcStatus KVC_CALL Manifest(const KvcObject* object, const KvcBlock** blocks, uint64_t* count) {
  return ObjectManifest(object, blocks, count);
}
KvcStatus KVC_CALL MemoryInfo(const KvcObject* object, const KvcMemoryInfo** info) {
  return ObjectMemoryInfo(object, info);
}
void KVC_CALL Abandon(KvcObject* object) { AbandonWrite(object); }

KvcStatus KVC_CALL Poll(KvcObject* object, KvcTransferStatus* out) {
  return PollTransfer(object, out);
}
KvcStatus KVC_CALL Wait(KvcObject* object, uint64_t timeout) {
  return WaitTransfer(object, timeout);
}

#define TABLE_HEADER(T) sizeof(T), KVC_ABI_MAJOR, KVC_ABI_MINOR

const KvcProviderApi provider_api = {TABLE_HEADER(KvcProviderApi),
                                     Capabilities,
                                     Configure,
                                     Configuration,
                                     Shutdown,
                                     QueryExtension};
const KvcControl control_api = {
    TABLE_HEADER(KvcControl), Lookup, BeginWrite, Commit, Abort, QueryWrite, Remove};
const KvcData data_api = {TABLE_HEADER(KvcData), Import, Load, Store, nullptr};
const KvcObjectApi object_api = {
    TABLE_HEADER(KvcObjectApi), Retain, Release, Kind, Manifest, MemoryInfo, Abandon};
const KvcTransferApi transfer_api = {TABLE_HEADER(KvcTransferApi), Poll, Wait, nullptr};
const KvcPlugin plugin = {TABLE_HEADER(KvcPlugin),
                          {"kvc.host", 8},
                          Open,
                          &provider_api,
                          &control_api,
                          &data_api,
                          &object_api,
                          &transfer_api};

#undef TABLE_HEADER

}  // namespace
}  // namespace kvc_host

KVC_EXPORT KvcStatus KVC_CALL KvcGetProvider(uint32_t requested_major, uint32_t requested_minor,
                                             const KvcPlugin** out) {
  (void)requested_minor;
  if (!out) return KvcStatus{KVC_INVALID_ARGUMENT, {}, nullptr, nullptr};
  *out = nullptr;
  if (requested_major != KVC_ABI_MAJOR) return KvcStatus{KVC_UNSUPPORTED, {}, nullptr, nullptr};
  *out = &kvc_host::plugin;
  return KvcStatus{KVC_OK, {}, nullptr, nullptr};
}
