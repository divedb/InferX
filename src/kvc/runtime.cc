#include <atomic>
#include <chrono>
#include <cstring>
#include <thread>
#include <utility>

#include "kvc/cache.h"

#if defined(_WIN32)
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace kvc {
namespace {
void discard(KvcStatus raw) noexcept {
  if (raw.release_diagnostic) raw.release_diagnostic(raw.diagnostic_owner);
}
Status consume(KvcStatus raw) {
  struct Cleanup {
    KvcStatus raw;
    ~Cleanup() {
      if (raw.release_diagnostic) raw.release_diagnostic(raw.diagnostic_owner);
    }
  } cleanup{raw};
  return {raw.code, raw.diagnostic.data ? std::string(raw.diagnostic.data, raw.diagnostic.size)
                                        : std::string()};
}
Status stale() { return {KVC_STALE_HANDLE, "Invalid or foreign-session handle"}; }
Status Unsupported() { return {KVC_UNSUPPORTED, "Optional operation is unsupported"}; }
template <class T>
bool TableValid(const T* table) {
  return table && table->struct_size >= sizeof(T) && table->abi_major == KVC_ABI_MAJOR;
}
bool Valid(const KvcPlugin* p) {
  return TableValid(p) && p->provider_id.data && p->provider_id.size && p->open &&
         TableValid(p->provider) && TableValid(p->control) && TableValid(p->data) &&
         TableValid(p->object) && TableValid(p->transfer) && p->provider->capabilities &&
         p->provider->configure && p->provider->configuration && p->provider->shutdown &&
         p->control->lookup && p->control->begin_write && p->control->commit &&
         p->control->abort && p->control->query_write && p->control->remove &&
         p->data->import_memory && p->data->load && p->data->store && p->object->retain &&
         p->object->release && p->object->kind && p->object->manifest &&
         p->object->memory_info && p->object->abandon_write && p->transfer->poll &&
         p->transfer->wait;
}
}  // namespace

namespace detail {
struct Module {
#if defined(_WIN32)
  HMODULE library = nullptr;
#else
  void* library = nullptr;
#endif
  const KvcPlugin* api = nullptr;
  std::atomic<bool> unload{true};
  ~Module() {
    if (!library || !unload.load()) return;
#if defined(_WIN32)
    FreeLibrary(library);
#else
    dlclose(library);
#endif
  }
};
struct Session {
  std::shared_ptr<Module> module;
  KvcObject* object = nullptr;
  const KvcPlugin* api() const { return module->api; }
  ~Session() {
    if (object) {
      auto result = api()->provider->shutdown(object, 0);
      const bool drained = result.code == KVC_OK;
      discard(result);
      if (drained) {
        api()->object->release(object);
      } else {
        // Dropping the last facade/transfer does not cancel accepted work.
        // Keep the session and DSO alive until provider-internal work drains.
        try {
          std::thread([module = module, object = object] {
            for (;;) {
              auto result = module->api->provider->shutdown(object, 0);
              bool drained = result.code == KVC_OK;
              discard(result);
              if (drained) break;
              std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
            module->api->object->release(object);
          }).detach();
        } catch (...) {
          // If even the reaper cannot be started, retain the session and code
          // for process lifetime rather than unload beneath accepted work.
          module->unload.store(false);
        }
      }
    }
  }
};
class Access {
 public:
  template <class T>
  static T adopt(std::shared_ptr<Session> session, KvcObject* object) {
    T result;
    result.session_ = std::move(session);
    result.object_ = object;
    return result;
  }
  static bool belongs(const Object& object, const std::shared_ptr<Session>& session) {
    return object.Valid() && object.session_ == session;
  }
};
Object::Object(std::shared_ptr<detail::Session> s, KvcObject* p)
    : session_(std::move(s)), object_(p) {}
Object::Object(const Object& other) : session_(other.session_), object_(other.object_) {
  if (object_) session_->api()->object->retain(object_);
}
Object& Object::operator=(const Object& other) {
  if (this != &other) {
    Object copy(other);
    *this = std::move(copy);
  }
  return *this;
}
Object::Object(Object&& other) noexcept
    : session_(std::move(other.session_)), object_(std::exchange(other.object_, nullptr)) {}
Object& Object::operator=(Object&& other) noexcept {
  if (this != &other) {
    Reset();
    session_ = std::move(other.session_);
    object_ = std::exchange(other.object_, nullptr);
  }
  return *this;
}
void Object::Reset() noexcept {
  if (object_) session_->api()->object->release(object_);
  object_ = nullptr;
  session_.reset();
}
Object::~Object() { Reset(); }
}  // namespace detail

WriteHandle::WriteHandle(WriteHandle&& other) noexcept : Object(std::move(other)) {}
WriteHandle& WriteHandle::operator=(WriteHandle&& other) noexcept {
  if (this != &other) {
    if (object_) session_->api()->object->abandon_write(object_);
    Object::operator=(std::move(other));
  }
  return *this;
}
WriteHandle::~WriteHandle() {
  if (object_) session_->api()->object->abandon_write(object_);
}

namespace {
Result<std::span<const KvcBlock>> Manifest(const std::shared_ptr<detail::Session>& session,
                                           KvcObject* object) {
  if (!object) return std::unexpected(stale());
  const KvcBlock* blocks = nullptr;
  uint64_t count = 0;
  auto s = consume(session->api()->object->manifest(object, &blocks, &count));
  if (!s.Ok()) return std::unexpected(std::move(s));
  return std::span<const KvcBlock>(blocks, count);
}
Result<WriteStatus> ToWriteStatus(KvcStatus raw, const KvcWriteStatus& result) {
  auto s = consume(raw);
  if (!s.Ok()) return std::unexpected(std::move(s));
  WriteStatus status{result.state, {}};
  if (result.block_count)
    status.blocks.assign(result.blocks, result.blocks + result.block_count);
  return status;
}
}  // namespace
Result<std::span<const KvcBlock>> ReadHandle::Blocks() const {
  return Manifest(session_, object_);
}
Result<std::span<const KvcBlock>> WriteHandle::Blocks() const {
  return Manifest(session_, object_);
}
Result<KvcMemoryInfo> MemoryRegion::Info() const {
  if (!Valid()) return std::unexpected(stale());
  const KvcMemoryInfo* info = nullptr;
  auto s = consume(session_->api()->object->memory_info(object_, &info));
  if (!s.Ok()) return std::unexpected(std::move(s));
  return *info;
}
Result<TransferStatus> TransferHandle::Poll() const {
  if (!Valid()) return std::unexpected(stale());
  KvcTransferStatus result{};
  result.struct_size = sizeof(result);
  auto s = consume(session_->api()->transfer->poll(object_, &result));
  if (!s.Ok()) return std::unexpected(std::move(s));
  return TransferStatus{result.state, consume(result.result)};
}
Status TransferHandle::Wait(Timeout timeout) const {
  return Valid() ? consume(session_->api()->transfer->wait(object_, timeout)) : stale();
}
Status TransferHandle::Cancel() const {
  if (!Valid()) return stale();
  auto fn = session_->api()->transfer->cancel;
  return fn ? consume(fn(object_)) : Unsupported();
}
Result<LookupResult> KVCacheControl::Lookup(const KvcLookupRequest& request, Timeout t) const {
  KvcLookupResult result{};
  result.struct_size = sizeof(result);
  auto s = consume(session_->api()->control->lookup(session_->object, &request, t, &result));
  if (!s.Ok()) return std::unexpected(std::move(s));
  return LookupResult{result.prefix, detail::Access::adopt<ReadHandle>(session_, result.read)};
}
Result<WriteHandle> KVCacheControl::BeginWrite(const KvcWriteRequest& request,
                                               Timeout t) const {
  KvcObject* object = nullptr;
  auto s =
      consume(session_->api()->control->begin_write(session_->object, &request, t, &object));
  if (!s.Ok()) return std::unexpected(std::move(s));
  return detail::Access::adopt<WriteHandle>(session_, object);
}
Result<WriteStatus> KVCacheControl::Commit(WriteHandle& write, Timeout t) const {
  if (!detail::Access::belongs(write, session_)) return std::unexpected(stale());
  KvcWriteStatus result{};
  result.struct_size = sizeof(result);
  auto raw =
      session_->api()->control->commit(session_->object, write.NativeHandle(), t, &result);
  return ToWriteStatus(raw, result);
}
Status KVCacheControl::Abort(const KvcTransactionId& id, Timeout t) const {
  return consume(session_->api()->control->abort(session_->object, &id, t));
}
Result<WriteStatus> KVCacheControl::QueryWrite(const KvcTransactionId& id, Timeout t) const {
  KvcWriteStatus result{};
  result.struct_size = sizeof(result);
  auto raw = session_->api()->control->query_write(session_->object, &id, t, &result);
  return ToWriteStatus(raw, result);
}
Status KVCacheControl::Remove(const KvcDigest& key, Timeout t) const {
  return consume(session_->api()->control->remove(session_->object, &key, t));
}

namespace {
struct AllocationOwner {
  std::atomic<uint64_t> refs{1};
  std::shared_ptr<void> allocation;
  static void KVC_CALL retain(void* context) {
    static_cast<AllocationOwner*>(context)->refs.fetch_add(1, std::memory_order_relaxed);
  }
  static void KVC_CALL release(void* context) {
    auto* owner = static_cast<AllocationOwner*>(context);
    if (owner->refs.fetch_sub(1, std::memory_order_acq_rel) == 1) delete owner;
  }
};
}  // namespace
Result<MemoryRegion> KVCacheData::ImportMemory(const KvcMemoryInfo& info,
                                               std::shared_ptr<void> owner) const {
  if (!owner)
    return std::unexpected(Status{KVC_INVALID_ARGUMENT, "Allocation owner is required"});
  auto* allocation = new AllocationOwner{{1}, std::move(owner)};
  KvcMemoryImport request{
      sizeof(request), info, {allocation, AllocationOwner::retain, AllocationOwner::release}};
  KvcObject* object = nullptr;
  auto raw = session_->api()->data->import_memory(session_->object, &request, &object);
  AllocationOwner::release(allocation);
  auto s = consume(raw);
  if (!s.Ok()) return std::unexpected(std::move(s));
  return detail::Access::adopt<MemoryRegion>(session_, object);
}
Result<TransferHandle> KVCacheData::Load(const ReadHandle& read,
                                         const KvcTransferRequest& r) const {
  if (!detail::Access::belongs(read, session_)) return std::unexpected(stale());
  KvcObject* object = nullptr;
  auto s =
      consume(session_->api()->data->load(session_->object, read.NativeHandle(), &r, &object));
  if (!s.Ok()) return std::unexpected(std::move(s));
  return detail::Access::adopt<TransferHandle>(session_, object);
}
Result<TransferHandle> KVCacheData::Store(WriteHandle& write,
                                          const KvcTransferRequest& r) const {
  if (!detail::Access::belongs(write, session_)) return std::unexpected(stale());
  KvcObject* object = nullptr;
  auto s = consume(
      session_->api()->data->store(session_->object, write.NativeHandle(), &r, &object));
  if (!s.Ok()) return std::unexpected(std::move(s));
  return detail::Access::adopt<TransferHandle>(session_, object);
}
Result<MappedRead> KVCacheData::MapRead(const ReadHandle& read, const KvcSpan& span) const {
  if (!detail::Access::belongs(read, session_)) return std::unexpected(stale());
  auto fn = session_->api()->data->map_read;
  if (!fn) return std::unexpected(Unsupported());
  KvcMappedRead raw{};
  raw.struct_size = sizeof(raw);
  auto s = consume(fn(session_->object, read.NativeHandle(), &span, &raw));
  if (!s.Ok()) return std::unexpected(std::move(s));
  MappedRead result;
  result.readiness_ = detail::Access::adopt<TransferHandle>(session_, raw.readiness);
  // Hold the module while invoking provider-owned release callbacks.
  result.owner_ = std::shared_ptr<void>(
      raw.owner.context,
      [owner = raw.owner, session = session_](void*) { owner.release(owner.context); });
  for (uint64_t i = 0; i < raw.binding_count; ++i) {
    session_->api()->object->retain(raw.bindings[i].region);
    auto region = detail::Access::adopt<MemoryRegion>(session_, raw.bindings[i].region);
    result.bindings_.push_back({raw.bindings[i], std::move(region), result.owner_});
  }
  return result;
}

KVCacheProvider::~KVCacheProvider() {
  discard(session_->api()->provider->shutdown(session_->object, 0));
}
std::string_view KVCacheProvider::Id() const {
  auto id = session_->api()->provider_id;
  return {id.data, id.size};
}
Result<Capabilities> KVCacheProvider::GetCapabilities() const {
  KvcCapabilities raw{};
  raw.struct_size = sizeof(raw);
  auto s = consume(session_->api()->provider->capabilities(session_->object, &raw));
  if (!s.Ok()) return std::unexpected(std::move(s));
  Capabilities result{raw.features, raw.memory_types, raw.layouts, {}};
  for (uint64_t i = 0; i < raw.schema_count; ++i) {
    const auto& schema = raw.schemas[i];
    result.schemas.push_back(
        {std::string(schema.name.data, schema.name.size), schema.major, schema.minor});
  }
  return result;
}
Status KVCacheProvider::Configure(const KvcConfig& config, Timeout t) {
  return consume(session_->api()->provider->configure(session_->object, &config, t));
}
Result<const KvcConfig*> KVCacheProvider::Configuration() const {
  const KvcConfig* result = nullptr;
  auto s = consume(session_->api()->provider->configuration(session_->object, &result));
  if (!s.Ok()) return std::unexpected(std::move(s));
  return result;
}
Result<Extension> KVCacheProvider::QueryExtension(const KvcSchema& schema) const {
  auto fn = session_->api()->provider->query_extension;
  if (!fn) return std::unexpected(Unsupported());
  KvcExtension result{};
  result.struct_size = sizeof(result);
  auto s = consume(fn(session_->object, &schema, &result));
  if (!s.Ok()) return std::unexpected(std::move(s));
  auto extension = detail::Access::adopt<Extension>(session_, result.owner);
  extension.table_ = result.table;
  return extension;
}
Status KVCacheProvider::Shutdown(Timeout t) {
  return consume(session_->api()->provider->shutdown(session_->object, t));
}
Result<std::unique_ptr<KVCacheProvider>> OpenProvider(const ProviderOpen& request, Timeout t) {
  if (request.library_path.empty() || request.library_path.find('\0') != std::string::npos)
    return std::unexpected(Status{KVC_INVALID_ARGUMENT, "A nonempty library path is required"});
  auto module = std::make_shared<detail::Module>();
#if defined(_WIN32)
  module->library = LoadLibraryA(request.library_path.c_str());
  if (!module->library) return std::unexpected(Status{KVC_UNAVAILABLE, "LoadLibrary failed"});
  auto entry = reinterpret_cast<KvcGetProviderFn>(
      GetProcAddress(module->library, KVC_PROVIDER_ENTRY_SYMBOL));
#else
  module->library = dlopen(request.library_path.c_str(), RTLD_NOW | RTLD_LOCAL);
  if (!module->library) return std::unexpected(Status{KVC_UNAVAILABLE, dlerror()});
  auto entry =
      reinterpret_cast<KvcGetProviderFn>(dlsym(module->library, KVC_PROVIDER_ENTRY_SYMBOL));
#endif
  if (!entry)
    return std::unexpected(Status{KVC_UNSUPPORTED, "Missing KVC provider entry point"});
  auto s = consume(entry(KVC_ABI_MAJOR, KVC_ABI_MINOR, &module->api));
  if (!s.Ok()) return std::unexpected(std::move(s));
  if (!Valid(module->api))
    return std::unexpected(
        Status{KVC_UNSUPPORTED, "Incompatible or incomplete KVC ABI tables"});
  auto session = std::make_shared<detail::Session>();
  session->module = std::move(module);
  s = consume(session->api()->open(&request.options, t, &session->object));
  if (!s.Ok()) return std::unexpected(std::move(s));
  if (!session->object || session->api()->object->kind(session->object) != KVC_SESSION)
    return std::unexpected(Status{KVC_INTERNAL, "Provider did not return a session"});
  return std::unique_ptr<KVCacheProvider>(new KVCacheProvider(std::move(session)));
}
}  // namespace kvc
