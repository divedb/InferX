# KV-cache provider extensions

InferX now separates engine cache allocation from external cache providers.
`include/kvc/abi/` is the provider SDK, `include/kvc/cache.h` is the consumer
C++ facade, and `plugins/kvcache/` contains independently built providers.
`adapters/inferx/` translates InferX allocation ownership into imported memory.
The SDK does not depend on InferX tensors, Abseil, CUDA, or a scheduler.

This is the initial, experimental implementation of `kvcache_plan.md`, packaged
as SDK 0.1. ABI tables use the version-1 layout; this is not a claim that the entire
proposed standard or its schema registry is finalized. The implementation
currently supports the fixed opaque-component profile described below. The
engine's existing `KvCache` and `KvBlockPool` remain its compute allocations;
inference scheduling is not automatically routed through a provider.

## Build and install the SDK

```sh
cmake -S . -B build \
  -DINFERX_BUILD_ENGINE=OFF \
  -DCMAKE_INSTALL_PREFIX=/tmp/kvc-sdk
cmake --build build -j
ctest --test-dir build --output-on-failure
cmake --install build
```

The normal engine build also builds the SDK. Set
`INFERX_BUILD_KVCACHE_PLUGINS=OFF` to omit bundled providers. CMake consumers use
`find_package(KvCache 0.1 CONFIG REQUIRED)`: `kvc::abi` supplies C headers and
`kvc::runtime` supplies the C++23 facade/loader. Only `include/kvc/` is installed
as the provider SDK. Installed providers live in `lib/kvcache/providers` (or the
platform's configured `CMAKE_INSTALL_LIBDIR`).

The standalone C provider-author scaffold can be built using only that install:

```sh
cmake -S examples/external_provider -B /tmp/external-provider \
  -DCMAKE_PREFIX_PATH=/tmp/kvc-sdk
cmake --build /tmp/external-provider -j
ctest --test-dir /tmp/external-provider --output-on-failure
```

The scaffold implements discovery and session ownership; its cache operations
explicitly return `Unsupported`. The functional host backend is the reference
for implementing those operations. External providers need not be copied into
InferX's source tree, and their dependencies belong to their own build targets.

## Public interfaces and loading

A provider exports the C symbol `KvcGetProvider`. It receives the requested
ABI major/minor and returns immutable `KvcPlugin` tables. It must reject an
incompatible major. Providers supporting newer minor versions retain the known
prefix of every table. The loader checks table sizes, major versions, and all
required function pointers before calling `open`.

| Table | Responsibilities |
| --- | --- |
| Plugin | Provider ID, factory, operation tables |
| Provider | Capabilities, configure once, immutable configuration, shutdown, optional extension query |
| Control | Lookup/acquire, begin, commit, abort, transaction query, retire |
| Data | Memory import, load, store, optional immutable mapping |
| Object | Retain/release, kind, immutable manifests, memory metadata, write abandonment |
| Transfer | Poll, wait, optional cancellation |

Consumer code opens a library explicitly:

```cpp
auto provider = kvc::OpenProvider({"/path/to/libkvc_host.so", {}});
if (!provider) { /* handle provider.error() */ }
// Inspect capabilities, configure, then use Control() and Data().
```

The loader uses local symbol visibility on POSIX and also has a Windows loader
path; Linux is the validated platform. There is no automatic directory scanning,
global registration constructor, or dependency on the current working directory
for discovery. Relative paths, if supplied, follow the platform loader's rules.

All C requests use fixed-width values, pointer/count arrays and opaque handles.
Extensible call records and tables start with `struct_size`; initialize it to
`sizeof(record)`. Named fixed-value records are immutable layouts; replacing
their shape requires a new versioned record/contract, not changing array stride.
Initialize reserved/unused fields to zero. Timeouts are nanoseconds;
`UINT64_MAX` means unbounded and zero means no waiting. Required operations must
be present. Optional table entries may be null or return `Unsupported`.

Capabilities describe possibilities, not every valid combination. Configuration
validates the complete request. Required features or extensions that cannot be
provided fail configuration. `Configuration()` exposes the accepted immutable
configuration; the current profile accepts it exactly or rejects it.

## Ownership and concurrency

- Request descriptors borrow memory during the call. Providers copy all metadata
  needed afterward. Capability/schema arrays and configuration pointers borrow
  the session; manifest pointers borrow their handle. Transaction-query manifests
  borrow the session. The facade copies status diagnostics, capabilities and
  transaction receipts into application-owned values.
- Every returned object owns one reference. Allocation owners have explicit C
  retain/release callbacks. The C++ facade bridges a `shared_ptr` locally; its
  representation never crosses the ABI. Release callbacks must not throw.
- A failed submission accepts no work and performs no asynchronous buffer access.
  An accepted operation returns a transfer handle and retains its dependencies,
  allocations, and cache objects until quiescence. Dropping a transfer does not
  cancel it. Terminal completion includes memory visibility and no further
  payload access; timeout or successful cancellation request alone does not.
- A read owns the exact generations acquired atomically with lookup. Retirement
  removes discoverability; existing readers remain valid. Mapped regions must
  independently retain that pin. `MappedBinding` copies retain their metadata.
- The write facade is move-only. Destruction calls nonblocking `abandon_write`
  before release, so internal transfer references cannot postpone abandonment.
  Providers drain accepted operations before reclaiming abandoned storage.
- Operations are safe concurrently on separate handles and transactions. The
  caller serializes store submissions and commit/abort on one transaction.
  Concurrent mutation/destruction of the same C++ wrapper is not supported.
- Closing rejects new work but permits completion observation, query, abort,
  and commit of admitted writes. `shutdown` returns `Busy` while owners remain;
  it must report success only when background work can no longer execute
  provider code (or release synchronously joins it). Objects retain the module.
  If the last wrapper disappears while shutdown still reports outstanding work,
  a runtime reaper retains the session/module and retries shutdown until drained.
  An unresponsive provider stays loaded; it is never forcibly unloaded.

The C++ operational methods return `std::expected<T, Status>`; local C++
allocation failures may throw. No exception may escape a provider C function.
Providers release their own allocated diagnostics and output memory.

## Host reference provider

The host provider is session-local, with a 64 MiB payload budget including
unpublished writes and retired blocks still pinned by readers. Transaction
metadata is retained for session lifetime. It has no persistent storage or
cross-session discovery. Configuration contains:

- Caller-supplied namespace and complete model-identity digests.
- Logical partition descriptor.
- Groups with unique IDs, positive tokens per block, and ascending unique layers.
- Group schema `kvc.opaque`, version 1.0, empty parameters.
- Components with unique IDs, layout `KVC_OPAQUE_BYTES`, schema `kvc.bytes`,
  version 1.0, empty parameters, and a positive fixed byte count per block.

The opaque schema treats component bytes as immutable checkpoints. The producer
and consumer must agree on their interpretation through the model/partition
identity. Adapters supply complete dependency digests and candidate resume plans;
the provider validates structural coverage and exact stored block records, but
cannot infer custom-state resume requirements from opaque bytes.

Blocks have a positive immutable valid-token count no greater than the group's
block size; the first token equals logical index times block size. Every selected
layer/component requires one complete binding. Layerwise operations are supported;
partial component transfers, tensors, scatter/gather subdivisions, mapping,
cancellation, native GPU synchronization and remote/device memory are rejected.
Imports require local CPU memory (`device_runtime="cpu"`, identifier `"0"`).
Copies complete synchronously and return terminal transfer handles. Dependencies
must belong to the same session; all this provider's transfers are already terminal.

Begin reserves the complete manifest privately. Commit publishes all its blocks
atomically only after every component of every layer was stored. Repeated commit
returns the same receipt. Invalid submissions leave the transaction unchanged;
failed *accepted* stores must poison transactions in providers supporting them.
Reusing a transaction ID for begin returns `Conflict`; use query to resolve an
uncertain begin/commit. Lookup evaluates all supplied candidates without assuming
availability is monotonic and returns the longest available candidate or a normal
miss. It rejects key matches whose block metadata differs.

Portable identity generation and numerical/schema compatibility remain adapter
responsibilities. This SDK does not yet implement the plan's proposed canonical
CBOR/SHA-256 identity helper or standard attention/MLA schema registry; callers
must not use pointer values, tensor shapes alone, or the test fixture keys as
production identities.

## CUDA tiered provider

`plugins/kvcache/cuda` implements `kvc.cuda`, a two-tier KV store: blocks are
GPU-resident while hot and written back to a host-memory tier under device
budget pressure. It targets the CPU-offload use case (plan section 27, item
12) with the same opaque-component profile as the host provider.

Storage and tiering:

- Each group's block payload is one device slot. Slots are `cudaMalloc`'d on
  demand from per-size-class free lists and recycled, not freed, so repeated
  publication and eviction do not fragment the device heap.
- `begin_write` reserves the manifest's bytes against the GPU budget
  (evicting committed state if needed); stores convert reservations into
  slots lazily, so an incomplete transaction costs nothing until it stores.
- When the GPU budget is exceeded, the least-recently-used committed,
  unpinned, GPU-resident block is copied synchronously to the host tier and
  its slot is recycled. When the host budget is exceeded, the LRU host block
  is retired from the index (a lookup miss) and its memory released.
- Loads that target device memory promote host-tier blocks back into GPU
  slots when budget allows, restoring GPU residency; otherwise they are
  served by direct host-to-device copies. Pins never move or drop data:
  blocks referenced by read handles or in-flight transfers are skipped as
  eviction victims, and a pinned block that cannot be evicted fails the
  reservation with `ResourceExhausted` rather than dropping data.

Transfers:

- Device-to-device and staged host copies are enqueued on one internal CUDA
  stream; completion is reported through per-transfer CUDA events, so
  `poll`/`wait` observe real asynchronous state. Transfers touching host
  memory complete synchronously at submission.
- An accepted transfer retains its read/write handle blocks, regions, and
  dependencies until terminal. Releasing a pending handle drains it (the
  work is already enqueued); dropping a handle never cancels it.
- A store transfer that fails in flight poisons its transaction: commit and
  further stores report `Aborted`, and `query_write` shows the aborted
  state. Pending stores make commit return `Busy`.

Sessions accept an options descriptor, schema `kvc.cuda.options` version 1,
24 canonical little-endian bytes: `u64 gpu_budget` (0 = automatic: min of
free/4 and 2 GiB at configure time), `u64 host_budget` (0 = 1 GiB default),
`u32 device_ordinal`, and one reserved zero word. Imports accept local CPU
memory (`cpu`/`0`, pageable or pinned) and device memory (`cuda`/ordinal)
matching the session's device. Mapping and cancellation are not advertised.

## Statistics extension

Providers may implement the optional `kvc.stats` extension (version 1.0,
declared in `kvc/extensions/stats.h`): lifetime counters for lookups, hits,
committed/retired/evicted blocks, transactions, stores and loads with byte
totals, device-slot allocations versus free-list reuses, GPU-to-host
offloads and host-to-GPU promotions with byte totals, and current tier
occupancy against the configured budgets. Both bundled providers implement
it; the async test fixture does not (the extension is optional, and
consumers must handle `Unsupported`). The counter set deliberately covers
the plan's observability gap (section 27, item 36) for single sessions.

## The run-bench command

`inferx run-bench` drives a loaded model end-to-end through a provider
session and reports cache behavior:

```sh
inferx run-bench --model models/Qwen3-0.6B --device cuda \
  --gpu-budget 8MiB --prompt-b "Second prompt..." --iterations 4
```

Each request looks the prompt prefix up in the provider; a hit loads the
pinned blocks into the engine KvCache and generation resumes after the
cached tokens, recomputing only the tail (and the final prompt token when
the whole prompt is cached). Computed prefixes publish block-by-block, so
device budgets smaller than the working set force LRU offload to the host
tier, and later hits restore (promote) blocks back to the device. A
`--prompt-b` alternates two prompts so offload and restore both occur under
pressure.

Every run first executes uncached baselines twice: greedy decoding should
be reproducible, and the run reports whether it is. When it is (CPU
backend), cached runs must reproduce the baseline tokens exactly and the
benchmark fails otherwise. Backends whose kernels are not bitwise
reproducible (observed with the 28-layer CUDA path; identical uncached
runs diverge) fall back to comparing the resumed prefill logits against
full recomputation, reporting the maximum difference and first-token
agreement, so cache infidelity is still distinguishable from kernel noise.

The final report dumps the `kvc.stats` record: hit rate and blocks covered
by hits, committed/retired/evicted blocks, transactions, store/load counts
and bytes, device-slot allocation versus reuse, GPU/CPU tier movement in
both directions, and current occupancy against both budgets. Sustained
misses on repeated prompts, offloads without matching promotions, or
occupancy above the configured budgets are the quickest signals of a
misbehaving cache.

## InferX integration boundary

Link `inferx::kvc_adapter` and use `inferx::ImportProviderMemory` to import owned
`Storage`. `inferx/cache/kv_bridge.h` builds on it: `KvCacheBridge` binds a
dense model's per-layer K/V tensors to provider blocks, derives prefix
identities from a token chain, publishes computed prompt KV block-by-block,
and loads pinned hits back so `Forward` resumes at the cached length. It retains the storage while regions/transfers need it, rejects borrowed
storage, and describes CPU/CUDA/HIP/Ascend allocations without dereferencing device
addresses. The selected provider decides whether the memory kind is supported.
Allocator lifetime and producer/consumer synchronization remain engine obligations.

The adapter intentionally does not expose `KvBlockPool`, block tables, requests,
scheduler state, streams, RDMA registrations, or backend allocation objects to
providers. Scheduling/reuse planning integration is a subsequent engine feature.

Optional vendor extensions use namespaced schema/version queries returning a
versioned table and a retained owner. The C++ `Extension` keeps both the owner and
provider code alive. Each extension must specify table/context use, ownership,
synchronization and negotiation; unknown IDs return `Unsupported`. There are no
standard optional extension tables published yet. Provider-private transports,
indexes, eviction and allocation do not need public plugin interfaces.

## Validation

`tests/kvc` checks C header compatibility, loader rejection, profile negotiation,
publication completeness, idempotent commit, generation pinning after retirement,
binding coverage/alias rejection, foreign-session rejection, memory ownership,
shutdown, and delayed provider cleanup after the last facade disappears.
`kvc_provider_conformance` runs the same behavioral suite against the host
provider, the CUDA provider, and an async fixture whose transfers complete on a
background thread with failure injection (pending polls, wait deadlines, busy
commits, poisoned transactions, failed dependencies, drain-after-drop).
`kvc_cuda_provider_test` exercises the tiered behavior on a real GPU: device
round trips, LRU offload, eviction to misses, promotion back to device memory,
pin interactions, dependent device transfers, and deterministic small budgets
set through the provider options. An engine-linked test checks the `Storage`
ownership bridge for CPU and CUDA allocations. These are conformance tests for
the implemented profile, not certification of all future kvc schemas.
