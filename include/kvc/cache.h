// Consumer-side C++ facade over the KV-cache provider ABI.
//
// Two sides exist, and they are deliberately different:
//
//   - Providers (plugins) implement the C function tables declared in
//     kvc/abi/provider.h (KvcPlugin and its five tables) and export
//     KvcGetProvider. Nothing in this header is implemented or subclassed
//     by a provider.
//   - Consumers (the engine, adapters, tests) use the kvc:: classes below
//     to load a provider library and drive it with RAII ownership and
//     std::expected error handling.
//
// Only the C ABI crosses the library boundary. Every handle below owns one
// provider object reference obtained through that ABI, and the runtime
// (kvc::runtime) keeps the provider library loaded while any reference to
// its objects survives. Local C++ allocation failures may throw; provider
// failures are reported as Status.

#ifndef KVC_CACHE_H_
#define KVC_CACHE_H_

#include <expected>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "kvc/abi/provider.h"

namespace kvc {

/// \brief Consumer-side operation result: a KvcCode value plus a copied
/// diagnostic string. code == KVC_OK means success.
struct Status {
  uint32_t code = KVC_OK;
  std::string diagnostic;
  /// \brief True when code is KVC_OK.
  bool Ok() const noexcept { return code == KVC_OK; }
};

/// \brief Result type for all facade operations: the value on success,
/// kvc::Status on failure.
template <class T>
using Result = std::expected<T, Status>;

/// \brief Timeout in nanoseconds. KVC_TIMEOUT_INFINITE is unbounded; zero
/// means do not wait.
using Timeout = uint64_t;

namespace detail {
struct Session;
class Access;

// Internal lifetime implementation shared by every handle type; not a
// provider inheritance interface. Keeps the owning session (and through it
// the loaded library) alive for as long as the wrapped object reference.
class Object {
 public:
  Object() = default;
  Object(const Object&);
  Object& operator=(const Object&);
  Object(Object&&) noexcept;
  Object& operator=(Object&&) noexcept;
  ~Object();
  /// \brief True while the handle wraps a live provider object.
  bool Valid() const noexcept { return object_ != nullptr; }
  // Borrowed, for constructing ABI requests. Never release it yourself.
  KvcObject* NativeHandle() const noexcept { return object_; }

 protected:
  Object(std::shared_ptr<detail::Session>, KvcObject*);
  void Reset() noexcept;
  std::shared_ptr<detail::Session> session_;
  KvcObject* object_ = nullptr;
  friend class detail::Access;
};
}  // namespace detail

/// \brief Pinned set of committed blocks returned by a successful lookup.
/// Copies retain; releasing the last copy unpins the state, allowing
/// eviction. Blocks are borrowed from the provider.
class ReadHandle final : public detail::Object {
 public:
  using Object::Object;
  /// \brief The pinned blocks, in logical order; borrowed from the handle.
  Result<std::span<const KvcBlock>> Blocks() const;
};

/// \brief Exclusive ownership of one unpublished write transaction,
/// obtained from BeginWrite. Move-only: destroying or moving away from it
/// abandons the transaction without waiting for pending stores.
class WriteHandle final : public detail::Object {
 public:
  using Object::Object;
  WriteHandle() = default;
  WriteHandle(const WriteHandle&) = delete;
  WriteHandle& operator=(const WriteHandle&) = delete;
  WriteHandle(WriteHandle&&) noexcept;
  WriteHandle& operator=(WriteHandle&&) noexcept;
  ~WriteHandle();
  /// \brief The transaction's reserved manifest; borrowed from the handle.
  Result<std::span<const KvcBlock>> Blocks() const;
};

/// \brief Retained access to caller-owned memory imported into the
/// provider (from ImportMemory). Keeps the allocation owner referenced.
class MemoryRegion final : public detail::Object {
 public:
  using Object::Object;
  /// \brief The stable memory description. Nested metadata borrows this
  /// owner.
  Result<KvcMemoryInfo> Info() const;
};

/// \brief Consumer-side transfer outcome: a KvcTransferState value plus,
/// once terminal, the transfer's Status.
struct TransferStatus {
  uint32_t state;
  Status result;
};

/// \brief Completion handle for an accepted load, store, or mapped read.
/// The transfer keeps its dependencies alive; dropping it does not cancel
/// it.
class TransferHandle final : public detail::Object {
 public:
  using Object::Object;
  /// \brief Reports current state without waiting.
  Result<TransferStatus> Poll() const;
  /// \brief Waits up to the timeout for a terminal state.
  Status Wait(Timeout timeout = KVC_TIMEOUT_INFINITE) const;
  /// \brief Requests cancellation; Unsupported when the provider lacks the
  /// Cancellation feature.
  Status Cancel() const;
};

/// \brief Lookup outcome: the matched prefix and a pinned ReadHandle. On a
/// normal miss, read is invalid and prefix carries no hit.
struct LookupResult {
  KvcPrefix prefix{};
  ReadHandle read;  // Invalid on a normal miss.
};

/// \brief Transaction state plus, once committed, a copied receipt of the
/// published blocks.
struct WriteStatus {
  uint32_t state;
  std::vector<KvcBlock> blocks;
};

/// \brief Provider abilities reported by GetCapabilities, with schemas
/// copied into owned strings.
struct Capabilities {
  uint64_t features;      ///< OR of supported KvcFeature values.
  uint64_t memory_types;  ///< Bit N advertises memory type N.
  uint64_t layouts;       ///< Bit N advertises layout N.
  /// \brief One supported semantic schema.
  struct Schema {
    std::string name;
    uint32_t major;
    uint32_t minor;
  };
  std::vector<Schema> schemas;
};

/// \brief Vendor extension returned by QueryExtension: the extension's
/// table pointer plus the owner keeping it and the provider alive.
class Extension final : public detail::Object {
 public:
  /// \brief The extension table, or nullptr when invalid. Its contract is
  /// defined by the extension's schema.
  const void* Table() const noexcept { return Valid() ? table_ : nullptr; }

 private:
  const void* table_ = nullptr;
  friend class KVCacheProvider;
};

/// \brief One mapping of a (block, layer, component) onto pinned memory.
/// The region independently retains the immutable cache pin;
/// metadata_owner retains the mapping's nested arrays.
struct MappedBinding {
  KvcBinding binding{};  // Nested shape arrays retained by metadata_owner.
  MemoryRegion region;   // Independently retains the immutable cache pin.
  std::shared_ptr<void> metadata_owner;
};

/// \brief Result of MapRead: pinned bindings plus a readiness transfer
/// that gates first access. Requires the Mapping feature.
class MappedRead {
 public:
  /// \brief The mapped bindings.
  std::span<const MappedBinding> Bindings() const { return bindings_; }
  /// \brief Transfer that must complete before the memory is read.
  const TransferHandle& Readiness() const { return readiness_; }

 private:
  std::shared_ptr<void> owner_;
  std::vector<MappedBinding> bindings_;
  TransferHandle readiness_;
  friend class KVCacheData;
};

/// \brief Control-plane facade: discovery, transactions, publication, and
/// retirement. Obtained from KVCacheProvider::Control; copyable and
/// borrowed from the provider session.
class KVCacheControl {
 public:
  /// \brief Evaluates all candidates; the longest fully available one is
  /// returned as a pinned read handle, or a normal miss.
  Result<LookupResult> Lookup(const KvcLookupRequest&, Timeout = KVC_TIMEOUT_INFINITE) const;
  /// \brief Reserves a manifest privately under a fresh transaction id;
  /// nothing is visible to Lookup until Commit.
  Result<WriteHandle> BeginWrite(const KvcWriteRequest&, Timeout = KVC_TIMEOUT_INFINITE) const;
  /// \brief Publishes the transaction atomically once every selected
  /// component is stored; repeat calls return the same receipt.
  Result<WriteStatus> Commit(WriteHandle&, Timeout = KVC_TIMEOUT_INFINITE) const;
  /// \brief Rolls back an unfinished transaction by id.
  Status Abort(const KvcTransactionId&, Timeout = KVC_TIMEOUT_INFINITE) const;
  /// \brief Reports a transaction's state and receipt by id.
  Result<WriteStatus> QueryWrite(const KvcTransactionId&, Timeout = KVC_TIMEOUT_INFINITE) const;
  /// \brief Retires a block key: removes discoverability; existing pins
  /// stay valid until released.
  Status Remove(const KvcDigest&, Timeout = KVC_TIMEOUT_INFINITE) const;

 private:
  explicit KVCacheControl(std::shared_ptr<detail::Session> session)
      : session_(std::move(session)) {}
  std::shared_ptr<detail::Session> session_;
  friend class KVCacheProvider;
};

/// \brief Data-plane facade: memory import and movement of cached state.
/// Obtained from KVCacheProvider::Data.
class KVCacheData {
 public:
  /// \brief Adopts an allocation kept alive through owner (e.g. a
  /// shared_ptr to the engine's Storage); the provider retains it until
  /// the last dependent transfer completes.
  Result<MemoryRegion> ImportMemory(const KvcMemoryInfo&, std::shared_ptr<void> owner) const;
  /// \brief Copies cached state into memory bound to a read handle.
  Result<TransferHandle> Load(const ReadHandle&, const KvcTransferRequest&) const;
  /// \brief Copies caller memory into a write transaction's blocks.
  Result<TransferHandle> Store(WriteHandle&, const KvcTransferRequest&) const;
  /// \brief Maps immutable cached state for direct reads; Unsupported when
  /// the provider lacks the Mapping feature.
  Result<MappedRead> MapRead(const ReadHandle&, const KvcSpan&) const;

 private:
  explicit KVCacheData(std::shared_ptr<detail::Session> session)
      : session_(std::move(session)) {}
  std::shared_ptr<detail::Session> session_;
  friend class KVCacheProvider;
};

/// \brief Open request for OpenProvider: the provider library path and
/// optional provider-specific open parameters (borrowed during the call).
struct ProviderOpen {
  std::string library_path;
  KvcDescriptor options{};  // Borrowed during OpenProvider.
};

/// \brief One loaded provider session. Drives negotiation and yields the
/// Control and Data facades; not copyable (use OpenProvider again for
/// another session). Destroying the last reference shuts the session down.
class KVCacheProvider {
 public:
  KVCacheProvider(const KVCacheProvider&) = delete;
  KVCacheProvider& operator=(const KVCacheProvider&) = delete;
  ~KVCacheProvider();
  /// \brief The provider's identity string (e.g. "kvc.host").
  std::string_view Id() const;
  /// \brief Reports the provider's abilities.
  Result<Capabilities> GetCapabilities() const;
  /// \brief Validates and installs the session's one configuration;
  /// deep-copied by the provider.
  Status Configure(const KvcConfig&, Timeout = KVC_TIMEOUT_INFINITE);
  /// \brief The installed configuration; borrowed until session release.
  Result<const KvcConfig*> Configuration() const;
  /// \brief The control-plane facade.
  KVCacheControl Control() const { return KVCacheControl(session_); }
  /// \brief The data-plane facade.
  KVCacheData Data() const { return KVCacheData(session_); }
  /// \brief Looks up a vendor extension table by schema.
  Result<Extension> QueryExtension(const KvcSchema&) const;
  /// \brief Drains or fences the session; Busy while owners or accepted
  /// work remain.
  Status Shutdown(Timeout = KVC_TIMEOUT_INFINITE);

 private:
  explicit KVCacheProvider(std::shared_ptr<detail::Session> session)
      : session_(std::move(session)) {}
  std::shared_ptr<detail::Session> session_;
  friend Result<std::unique_ptr<KVCacheProvider>> OpenProvider(const ProviderOpen&, Timeout);
};

/// \brief Loads a provider library by path, negotiates the ABI, opens a
/// session, and returns the driving facade. Fails with Unavailable when
/// the library cannot be loaded and Unsupported when it lacks a
/// compatible entry point or complete tables.
Result<std::unique_ptr<KVCacheProvider>> OpenProvider(const ProviderOpen&,
                                                      Timeout = KVC_TIMEOUT_INFINITE);

}  // namespace kvc

#endif  // KVC_CACHE_H_
