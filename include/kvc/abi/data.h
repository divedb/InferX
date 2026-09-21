#ifndef KVC_ABI_DATA_H_
#define KVC_ABI_DATA_H_

#include "kvc/abi/basics.h"

#ifdef __cplusplus
extern "C" {
#endif

/// \brief Where cached-state bytes physically reside. Bit N of
/// KvcCapabilities::memory_types advertises memory type N.
enum KvcMemoryType {
  KVC_HOST = 1,       ///< CPU-addressable pageable memory.
  KVC_PINNED_HOST,    ///< Page-locked host memory usable by device DMA.
  KVC_DEVICE,         ///< Device-local memory (GPU or accelerator).
  KVC_REMOTE_HOST,    ///< Host memory of another node.
  KVC_REMOTE_DEVICE,  ///< Device memory of another node.
  KVC_FABRIC          ///< Network-attached fabric memory.
};

/// \brief Permitted access to an imported memory region.
enum KvcAccess {
  KVC_READ_ONLY = 1,  ///< May be read as a store source; not a load destination.
  KVC_READ_WRITE = 2  ///< May be used as a store source or load destination.
};

/// \brief How an imported region is located: by local address or by an
/// opaque external-memory descriptor.
enum KvcLocatorKind {
  KVC_LOCAL_ADDRESS = 1,   ///< local_address field names a pointer.
  KVC_EXTERNAL_MEMORY = 2  ///< external descriptor names the allocation.
};

/// \brief Description of one physical memory allocation. Memory is a
/// capability: device and remote memory need not be CPU-dereferenceable.
typedef struct KvcMemoryInfo {
  uint32_t struct_size;
  uint32_t memory_type;   ///< A KvcMemoryType value.
  uint32_t access;        ///< A KvcAccess value.
  uint32_t locator_kind;  ///< A KvcLocatorKind value.
  uint64_t byte_size;
  uintptr_t local_address;      ///< Valid when locator is LocalAddress.
  KvcString device_runtime;     ///< e.g. "cpu", "cuda", "hip", "ascend".
  KvcString device_identifier;  ///< Runtime-defined device name.
  KvcDescriptor external;       ///< Interop descriptor when locator is
                                ///< ExternalMemory.
} KvcMemoryInfo;

/// \brief Import request: memory description plus the owner keeping the
/// allocation alive. The provider retains it until the last region and
/// transfer using it are gone.
typedef struct KvcMemoryImport {
  uint32_t struct_size;
  KvcMemoryInfo info;
  KvcOwner allocation_owner;
} KvcMemoryImport;

/// \brief Addresses a sub-extent of one block.
typedef struct KvcSlice {
  KvcDigest key;
  uint64_t first_token;  ///< Relative to the block's valid extent.
  uint64_t token_count;
} KvcSlice;

/// \brief Selected state for a transfer: one group's layers over a set of
/// block slices.
typedef struct KvcSpan {
  uint32_t group;
  const uint32_t* layers;  ///< Unique layers of this group.
  uint64_t layer_count;
  const KvcSlice* blocks;
  uint64_t block_count;
} KvcSpan;

/// \brief Binding of one (block, layer, component) to a byte range of an
/// imported region. The binding set must exactly cover the span's selected
/// slots.
typedef struct KvcBinding {
  KvcSlice block;
  uint32_t layer;
  uint32_t component;
  KvcObject* region;  ///< Imported memory region.
  uint64_t byte_offset;
  uint64_t byte_count;
  uint32_t layout;               ///< A KvcLayout value.
  uint32_t dtype;                ///< Schema-defined fixed-width dtype code for tensors.
  const uint64_t* shape;         ///< Tensor bindings only.
  const uint64_t* byte_strides;  ///< Tensor bindings only.
  uint64_t rank;                 ///< Zero for opaque bytes.
} KvcBinding;

/// \brief Lifecycle state of an accepted transfer, observed through poll
/// and wait.
enum KvcTransferState {
  KVC_PENDING = 1,        ///< Accepted; not yet terminal.
  KVC_SUCCEEDED,          ///< Terminal success; memory is visible.
  KVC_FAILED,             ///< Terminal failure; outcome is final.
  KVC_TRANSFER_CANCELLED  ///< Terminal after successful cancellation.
};

/// \brief Data movement request: a span selection, its buffer bindings,
/// and transfers that must complete first (dependencies).
typedef struct KvcTransferRequest {
  uint32_t struct_size;
  KvcSpan selection;
  const KvcBinding* bindings;
  uint64_t binding_count;
  KvcObject* const* dependencies;  ///< Transfer handles, same session.
  uint64_t dependency_count;
} KvcTransferRequest;

/// \brief Terminal-or-not outcome of a transfer, from poll and wait.
typedef struct KvcTransferStatus {
  uint32_t struct_size;
  uint32_t state;    ///< A KvcTransferState value.
  KvcStatus result;  ///< Meaningful once state is terminal.
} KvcTransferStatus;

/// \brief Result of mapping immutable cached state: bindings whose regions
/// are pinned for the mapping's lifetime, plus a readiness transfer.
typedef struct KvcMappedRead {
  uint32_t struct_size;
  const KvcBinding* bindings;
  uint64_t binding_count;
  KvcObject* readiness;  ///< Transfer that gates first access.
  KvcOwner owner;        ///< Owns bindings and their region references.
} KvcMappedRead;

#ifdef __cplusplus
}
#endif
#endif  // KVC_ABI_DATA_H_
