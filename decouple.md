# Hardware backend decoupling

## Status

Design. No implementation has started. This document analyzes InferX's CUDA
coupling and defines the architecture for making the engine backend-agnostic,
targeting CUDA (reference), ROCm, and NPU platforms such as Ascend (CANN).

Relationship to existing documents:

- `tensor_parallel.md` defines the model-level TP abstraction. The two designs
  intersect at the model refactor (both rewrite model load/forward onto shared
  primitives) and at the communicator (backend enum). Sequencing is addressed
  in §8.
- `nccl.md` already demonstrates the pattern this document generalizes: the
  `Communicator` interface keeps `nccl.h` out of model code behind an opaque
  `void* stream` and a byte-array bootstrap token. That seam is the proof of
  concept for the rest of the system.

## 1. Current architecture and CUDA dependency analysis

### 1.1 The dependency map

Measured, not estimated (grep of CUDA/NCCL tokens across the tree):

```text
                CUDA-free today                 CUDA-coupled today
+---------------------------------------+  +--------------------------------+
| src/scheduler   (0 tokens)            |  | csrc/            6,466 lines   |
| src/server      (43 files, 0 tokens)  |  |   11 .cu files, 58 __global__  |
| src/api, src/observe, src/tokenizer   |  |   + gemm_cublaslt.cc (735 ln)  |
| src/support     (flag strings only)   |  | src/model/*.cc   (3 models +   |
| src/main        (a log string)        |  |   mla, moe_ffn, weight_loader) |
| core: dtype, shape, tensor, tensor_   |  | src/engine: kv_autosize.cc,    |
|   view, storage, arena, caching_      |  |   qwen_runner.cc               |
|   allocator, kv_cache, status         |  | src/comm/*.cc    (guarded)     |
| comm headers    (void* stream)        |  | core: device_buffer.cc,        |
| 72 of 105 public headers              |  |   allocator.cc   (guarded)     |
+---------------------------------------+  | 7 public headers include CUDA  |
                                           +--------------------------------+
```

The control plane (scheduler, server, API, tokenization, observability) is
already hardware-neutral. The coupling is confined to the data plane: kernels,
models, the engine's device-touching edges, and the guarded implementations in
core/comm. This is a favorable starting position — the work is deep but
narrow.

### 1.2 What "CUDA coupling" concretely means here

Five distinct dependency classes, which need different treatments:

1. **Device kernels** — 58 `__global__` kernels in `csrc/` (layers, MLA, MoE,
   quantize, gpt-oss, MXFP4/W4A16 GEMMs), written in CUDA C++ against
   `__nv_bfloat16`, `cuda_fp8.h`, `curand_kernel.h`.
2. **Vendor libraries** — cuBLASLt (`csrc/gemm_cublaslt.cc`, the only file
   including `cublasLt.h`), FlashInfer (header-only, wrapped in
   `flashinfer_{attention,prefill}.cu` behind pimpl classes), NCCL (confined
   to `src/comm/nccl_communicator.cc`). CUTLASS, FlashMLA, and hpc-ops are
   vendored but currently unreferenced by the build.
3. **Runtime services** — streams, events, graph capture/launch, async
   copies, pinned host memory, device selection, memory queries. Spread
   through `src/model/*.cc` (heavily), `weight_loader.cc`,
   `src/engine/{kv_autosize,qwen_runner}.cc`, `src/comm/*.cc`,
   `src/core/{device_buffer,allocator}.cc`.
4. **Type leakage** — `cudaStream_t` in the API of 13 public headers
   (7 include `cuda_runtime_api.h`; 6 more use the type relying on transitive
   includes — already fragile today). No public header leaks
   `cudaEvent_t`/`cudaGraph*`/`__nv_bfloat16`.
5. **Build machinery** — `cmake/InferXCuda.cmake` (toolchain floor, arch
   flags, `INFERX_WITH_CUDA`), `CUDA::cudart` linked into every target via
   `inferx::flags`, CMake-level exclusion (not `#ifdef`s) keeping unguarded
   model/engine sources out of host-only builds.

### 1.3 What is already right

Worth naming, because the design below extends these patterns rather than
inventing new ones:

- `DataType` (`core/dtype.h`) is fully vendor-neutral: bf16/f16 are
  `uint16_t` bit patterns on the host, and the `__nv_bfloat16` mapping exists
  only inside `.cu` files. FP8 FNUZ variants (the CDNA3/ROCm flavor) are
  already in the enum.
- `Communicator` proves the opaque-handle pattern: `void* stream`,
  fixed-size byte token instead of `ncclUniqueId`, capability struct instead
  of backend sniffing.
- `CublasLtGemm`, `FlashInferDecode`, `FlashInferPrefill` are pimpl classes
  that exist precisely to keep vendor headers out of `include/`.
- Error handling is centralized: one `CudaErrorToStatus` +
  `INFERX_CUDA_RETURN_IF_ERROR` (199 call sites) — a single conversion seam,
  not scattered ad-hoc handling.
- `-DINFERX_ENABLE_CUDA=OFF` already produces a working host-only build
  (core, comm, scheduler, server libs, loader half of model, unit tests).
- Tests are structured for hardware absence: `ctest -L unit` must pass on a
  GPU-less machine; device tests skip via `CudaAvailable()`.

## 2. Tightly coupled classes and modules

Responsibilities and coupling, per module:

| Module / class | Responsibility | Coupling |
|---|---|---|
| `csrc/` kernels + `kernels/*.h` | All device compute: norms, RoPE, activations, paged/MLA attention, MoE routing, sampling, quantization, quantized GEMMs | Total: CUDA C++ sources, `cudaStream_t` in 60+ API signatures |
| `kernels::CublasLtGemm` | Dense GEMM (bf16/f16/fp8) with per-shape plan cache | cuBLASLt behind pimpl; `cudaStream_t` in API |
| `kernels::FlashInferDecode/Prefill` | Production paged attention | FlashInfer (CUDA-only library) behind pimpl |
| `Qwen2Model`, `GptOssModel`, `DeepseekV2Model`, `MlaAttentionLayer`, `MoeFfn` | Model forward passes, weight upload, KV attach, graph capture, sampling readback | Direct: own `cudaStream_t`/`cudaEvent_t`, ~55 `cudaMemcpy*` sites, pinned buffers, per-model `cudaGraph*` capture/launch code (qwen2, gpt_oss) |
| `WeightLoader` | Pipelined checkpoint→device upload | Pinned staging ring: `cudaHostAlloc`, `cudaMemcpyAsync`, `cudaEvent*` (all `#ifdef`-guarded; CPU path exists) |
| `DeviceBuffer`, `AllocatorFor` | Device memory ownership | `cudaMalloc/Free/SetDevice` in guarded `.cc`; API clean |
| `src/engine/kv_autosize.cc` | KV pool sizing from free VRAM | `cudaMemGetInfo` (sole site) |
| `src/engine/qwen_runner.cc` | Rank workers | `cudaSetDevice` per worker thread |
| `comm` implementations | Collectives, timing | `cudaEvent*` timing ring, HostSim staging copies, NCCL calls (all guarded) |
| `core/cuda_utils.h` | Error conversion, device count | The CUDA-specific utility header itself |
| `DeviceId` (`core/device.h`) | Placement identity | `DeviceKind{kCpu,kCuda}` — no slot for other vendors; `Cuda(0)` defaults in 3 headers |
| `cmake/InferXCuda.cmake`, `csrc/CMakeLists.txt` | Toolchain | nvcc floor, arch list, cudart-for-everyone |

Explicitly **not** coupled and untouched by this design: scheduler, all of
`src/server`, api, tokenizer, observe, support, `ForwardBatch`, `ModelConfig`,
`Checkpoint`/safetensors, dtype/shape/tensor core types, prefix cache. GPU
telemetry is already out-of-process (DCGM exporter in docker-compose), so
observability needs no code changes per backend — only a different exporter
container.

## 3. Target architecture

### 3.1 Two boundaries, not one

The five dependency classes in §1.2 collapse into **two abstraction
boundaries** with different shapes:

```text
+-------------------------------------------------------------------+
| Control plane: server, scheduler, api, tokenizer, observe         |
|                (already neutral — no change)                      |
+-------------------------------------------------------------------+
| Engine + models + TP primitives (tensor_parallel.md)              |
|   - hold opaque Stream/Event handles, never vendor types          |
|   - call ops::* for compute, DeviceRuntime for services           |
+----------------------------------+--------------------------------+
| Boundary A: ops library          | Boundary B: device runtime     |
|   inferx::ops — the compute API  |   inferx::device::Runtime —    |
|   (norms, rope, attention, gemm, |   malloc/free, streams, events,|
|   moe, sampling, quantize, ...)  |   copies, pinned host memory,  |
|                                  |   graphs, device queries,      |
|                                  |   error→Status                 |
+----------------+---------------- +---------------+----------------+
| backends/cuda  | backends/rocm   | backends/     | backends/cpu   |
|  cuBLASLt,     |  hipBLASLt,     |  ascend       |  (host sim /   |
|  FlashInfer,   |  CK/aiter,      |  aclnn/ATB,   |   loader path, |
|  .cu kernels,  |  HIP-compiled   |  AscendC,     |   exists today)|
|  NCCL          |  kernels, RCCL  |  HCCL         |                |
+----------------+-----------------+---------------+----------------+
```

**Boundary A (ops)** is coarse-grained and semantic: "run RMSNorm", "run
paged decode attention", "GEMM y = x·Wᵀ". This is the level at which an NPU
backend can plug in vendor operator libraries (aclnn/ATB) that have nothing
in common with CUDA kernels. The existing `kernels/*.h` free functions are
already almost exactly this API — `Status f(TensorView..., stream)` — which
is why this design is a formalization, not a rewrite.

**Boundary B (runtime)** is fine-grained and mechanical: allocate, copy,
synchronize, record. Every vendor has an isomorphic API here (cudart / HIP /
aclrt), so a small interface covers all three.

The key architectural rule: **model and engine code may use only opaque core
types plus these two boundaries.** No vendor header, no vendor type, no
`#ifdef INFERX_WITH_<vendor>` in model/engine/scheduler/server sources.

### 3.2 Core type changes

#### Opaque execution handles

New in `core/` (names illustrative):

```cpp
// core/stream.h  (sketch)
// Opaque, trivially copyable, zero-cost wrappers. The pointee type is owned
// by the active backend (cudaStream_t / hipStream_t / aclrtStream / null).
struct Stream  { void* handle = nullptr; };
struct Event   { void* handle = nullptr; };
```

- Every `cudaStream_t` in a public kernel/model header becomes `Stream`.
  This also fixes the six headers that use `cudaStream_t` today without
  including any CUDA header.
- `Communicator::AllReduceSum(const TensorView&, void*)` migrates to
  `Stream` for uniformity (its `void*` was this pattern before it had a
  name).
- Backends cast at their edge: `static_cast<cudaStream_t>(s.handle)`. No
  virtual call, no overhead on the hot path.
- Stream/event *creation and destruction* is not on these structs — it lives
  on `DeviceRuntime`, because lifetime is a runtime service.

#### DeviceId

`DeviceKind` grows per-vendor values; the identity stays `(kind, index)`:

```cpp
enum class DeviceKind : uint8_t { kCpu = 0, kCuda = 1, kRocm = 2, kAscend = 3 };
bool IsAccelerator() const;   // kind != kCpu
```

Survey fact: all 35 `IsCuda()` call sites outside core mean "not host", so
they migrate mechanically to `IsAccelerator()`. The three `DeviceId::Cuda(0)`
default arguments in public headers (`kv_cache.h`, `weight_loader.h`,
`communicator.h`) become explicit parameters or a
`DeviceRuntime::default_device()` — defaults naming a vendor are exactly the
habit this design removes (the same cleanup `tensor_parallel.md` requires via
`ParallelContext::device()`).

#### What deliberately does not change

- `DataType` — already neutral, including FNUZ fp8 for CDNA3. Per-backend
  dtype *support* becomes a capability (an Ascend op library or an older HIP
  stack may not cover fp8e4m3; see §3.5).
- `TensorView`/`Tensor`/`Shape`/`Storage` — already neutral.
  `INFERX_HOST_DEVICE` in `support/host_device.h` extends its compiler
  condition (`__CUDACC__ || __HIP__ || ...`); it is inert for pure-host
  builds and for backends that never parse these headers in device code.
- `ForwardBatch`, `ModelConfig`, KV block-table logic — neutral.

### 3.3 Boundary B: the device runtime

```cpp
// core/device_runtime.h  (sketch)
class DeviceRuntime {
 public:
  virtual ~DeviceRuntime() = default;

  // Identity & discovery
  virtual DeviceKind kind() const = 0;
  virtual int device_count() const = 0;
  virtual Status SetDevice(DeviceId) = 0;          // thread-local, like today
  virtual StatusOr<MemInfo> GetMemInfo(DeviceId) = 0;   // kv_autosize

  // Memory
  virtual StatusOr<void*> Alloc(DeviceId, size_t bytes) = 0;
  virtual Status Free(DeviceId, void* p) = 0;
  virtual StatusOr<void*> AllocPinnedHost(size_t bytes) = 0;
  virtual Status FreePinnedHost(void* p) = 0;

  // Copies (enum CopyKind { kH2D, kD2H, kD2D })
  virtual Status Copy(void* dst, const void* src, size_t n, CopyKind) = 0;
  virtual Status CopyAsync(void* dst, const void* src, size_t n, CopyKind,
                           Stream) = 0;

  // Streams & events
  virtual StatusOr<Stream> CreateStream(DeviceId) = 0;   // non-blocking
  virtual Status DestroyStream(Stream) = 0;
  virtual Status SynchronizeStream(Stream) = 0;
  virtual StatusOr<Event> CreateEvent(bool timing) = 0;
  virtual Status DestroyEvent(Event) = 0;
  virtual Status RecordEvent(Event, Stream) = 0;
  virtual Status SynchronizeEvent(Event) = 0;
  virtual StatusOr<bool> QueryEvent(Event) = 0;
  virtual StatusOr<float> ElapsedMs(Event start, Event end) = 0;

  // Graph capture (capability-gated; see §3.5)
  virtual StatusOr<GraphHandle> BeginCapture(Stream) = 0;   // thread-local mode
  virtual StatusOr<GraphExec> EndCaptureAndInstantiate(Stream) = 0;
  virtual Status LaunchGraph(GraphExec, Stream) = 0;
  virtual Status DestroyGraph(GraphExec) = 0;

  virtual const RuntimeCapabilities& capabilities() const = 0;
};
```

Design points:

- **Virtual dispatch is acceptable here.** These calls are per-step or
  per-load, not per-token-per-layer: copies at batch upload/readback, event
  ops a handful per step, allocation at load/attach. The per-layer hot path
  (kernel launches) does not go through this interface — it goes through
  Boundary A.
- `DeviceBuffer` and `AllocatorFor` keep their public API and reroute their
  guarded internals through the runtime; `CachingAllocator`/`Arena` sit above
  and don't change.
- Error conversion moves inside each backend: `core/cuda_utils.h` becomes
  `backends/cuda/cuda_utils.h`; its 199 macro call sites live almost entirely
  in files that move into or behind backends anyway. Core keeps only
  `Status`.
- CPU implements the interface trivially (aligned alloc, memcpy, no-op
  streams, no graphs) — this *is* today's `#else` branches, promoted from
  preprocessor branches to a first-class backend, which also upgrades the
  host-only build from "compiles a subset" to "runs the same architecture."

### 3.4 Boundary A: the ops library

The 12 `kernels/*.h` headers already define a coherent op API. The design:

1. **Signatures stay free functions** in `inferx::ops` (renamed from
   `kernels` to mark the boundary), `Status f(const TensorView&..., Stream)`.
   Models keep calling them directly — no op-object graph, no registry lookup
   per launch.
2. **Resolution is link-time: exactly one device backend per binary** (plus
   CPU, always present). The build selects `backends/cuda` or
   `backends/rocm` or `backends/ascend`; each provides the `ops::*` symbols
   for accelerator tensors. This matches deployment reality (a server binary
   targets the machine's hardware), keeps zero dispatch overhead, avoids a
   premature plugin system, and mirrors how CMake already gates `csrc`.
   A runtime registry can be layered on later without changing any op
   signature — that door stays open, it just isn't paid for now.
3. **Stateful ops keep the pimpl pattern**, generalized:
   - `CublasLtGemm` → `ops::Gemm` (same API: `LinearBF16/F16/F8`, `Warm`,
     per-shape plan cache, one instance per rank thread). Backed by cuBLASLt
     / hipBLASLt / aclnn matmul.
   - `FlashInferDecode/Prefill` → `ops::PagedDecode` / `ops::PagedPrefill`
     (plan/run split preserved — the plan-buffer + run-on-stream shape is
     vendor-general and Ascend's ATB paged attention fits it). Backed by
     FlashInfer / CK-aiter / ATB.
4. **Op granularity is the contract, kernel granularity is not.** Backends
   are free to fuse or split internally (e.g. hpc-ops' fused
   allreduce+RMSNorm could later back a fused op on CUDA; ATB ops are
   naturally coarser). New fused ops enter the interface only when a model
   consumes them, with a composition-of-existing-ops reference semantics so
   every backend can implement them correctly by composition first.
5. **Numerics contract per op**: accumulation dtype (fp32), rounding, and a
   tolerance stated where exactness is impossible. This is what makes
   cross-backend differential testing (§7) meaningful rather than aspirational.

### 3.5 Capabilities instead of assumptions

Vendors differ in ways that must be queryable, not assumed. Extending the
`CommCapabilities` precedent:

```cpp
struct RuntimeCapabilities {
  bool graph_capture = false;       // CUDA yes; HIP yes (hipGraph);
                                    // Ascend initially no
  bool pinned_host_memory = false;
  bool device_sampling = false;     // curand-dependent path today
  DtypeSet supported_compute;       // fp8 e4m3 vs fnuz, mxfp4 support, ...
};
```

Consumers already exist: graph capture is gated on
`capabilities().cuda_graph_capture` (renamed `graph_capture`), DeepSeek-V2
already "serves without CUDA graphs by design," so the no-graph serving path
is exercised today. Quantized-weight paths (`FP8`, `INT4`, `MXFP4`) check
`supported_compute` at load and fail fast with a clear error instead of
mid-forward.

`CommBackend` gains `kRccl` and `kHccl`. `Communicator` needs no other
change — RCCL is API-identical to NCCL (the existing `nccl_communicator.cc`
compiles against it under HIP), and HCCL implements the same interface with
its own bootstrap token type behind the existing opaque-bytes pattern.

## 4. Backend designs

### 4.1 CUDA (reference backend)

`csrc/` and the CUDA-specific `.cc` internals move to `backends/cuda/`
essentially as-is: kernels, cuBLASLt Gemm, FlashInfer wrappers, the runtime
implementation (today's guarded code in `device_buffer.cc`, `allocator.cc`,
`weight_loader.cc`, model files), NCCL communicator, `cuda_utils.h`.
Behavior, performance, and the nvcc toolchain floor are unchanged. The CUDA
backend is the semantic reference: differential tests define correctness for
other backends against it (and against CPU reference ops where they exist —
`csrc/layers.cu`'s naive attention already plays this role for FlashInfer).

### 4.2 ROCm

ROCm is API-isomorphic to CUDA, which dictates a **shared-source strategy
inside the backend, not a shim in core**:

- **Runtime**: thin `DeviceRuntime` over HIP (hipMalloc, hipStream, hipEvent,
  hipGraph, hipHostMalloc, hipMemGetInfo). Mechanical.
- **Kernels**: the 6.5k lines of `.cu` compile under HIP with a portability
  header private to `backends/rocm/` (mapping `cuda*`→`hip*`,
  `__nv_bfloat16`→`hip_bfloat16`, curand→hiprand). Wavefront size (64 on
  CDNA vs 32) must be audited anywhere a kernel assumes warp=32
  (shuffles/reductions in sampling, MoE top-k, attention reference kernels) —
  this is the main correctness risk, and the per-op differential tests are
  the guard.
- **GEMM**: `ops::Gemm` over hipBLASLt (API-parallel to cuBLASLt, including
  the plan/heuristic model). FP8 uses the FNUZ types on CDNA3 — the dtype
  enum already distinguishes these; the loader's quantize path selects the
  variant from `supported_compute`.
- **Attention**: FlashInfer does not support ROCm. Options, in order:
  composable-kernel/aiter paged attention behind `ops::PagedDecode/Prefill`;
  fallback to the HIP-compiled reference paged-attention kernels (correct,
  slower) to unblock bring-up. The plan/run API shape hides the choice.
- **Comm**: RCCL through the existing NCCL code path (same symbols).

### 4.3 Ascend (CANN)

Ascend is *not* API-isomorphic — no CUDA-like kernel language toolchain
compatibility, different memory/stream semantics in places, operator
libraries as the primary compute interface. This is exactly why Boundary A is
op-granular:

- **Runtime**: `DeviceRuntime` over AscendCL (`aclrtMalloc`, `aclrtStream`,
  `aclrtEvent`, `aclrtMallocHost`, `aclrtMemcpyAsync`). The interface in §3.3
  was checked against aclrt's shape — it maps 1:1 except graphs.
- **Ops**: implemented against CANN operator libraries — aclnn for
  norms/activations/GEMM, ATB (Ascend Transformer Boost) for paged attention
  and fused transformer ops. Custom AscendC kernels only where no library op
  matches (candidates: sampling with penalties, MoE dispatch layout, MXFP4 —
  or those dtypes simply report unsupported initially).
- **Graphs**: `graph_capture = false` initially; the engine's existing
  no-graph path serves. CANN's graph facilities can be evaluated later behind
  the same `GraphExec` handles.
- **Comm**: HCCL communicator implementing `Communicator`; `kHccl` backend
  enum; bootstrap via HCCL's root-info bytes behind the opaque-token pattern.
- **KV cache**: block layout stays engine-owned (`KvBlockPool` is already
  neutral), but ATB attention ops constrain acceptable layouts/block sizes —
  the ops interface must let the backend *state* its preferred KV layout the
  way `MlaAttentionLayer::LayoutFor(config)` already does per-model. That
  existing hook generalizes to "backend × model chooses the layout."
- **Realism note**: Ascend support is a port of the compute strategy, not a
  recompile. The abstraction's job is to make that port additive (a new
  `backends/ascend/` directory) rather than invasive (edits across model and
  engine code).

### 4.4 Build system

- `INFERX_BACKEND` (string: `cuda` | `rocm` | `ascend` | `none`) replaces
  `INFERX_ENABLE_CUDA` as the primary switch (kept as a deprecated alias
  during migration). `none` is today's host-only build, now with the CPU
  runtime as a real backend.
- `cmake/InferXCuda.cmake` → `cmake/backends/Cuda.cmake` plus siblings; each
  backend file owns its toolchain detection, floor checks, arch flags, and
  vendor library discovery. `inferx::flags` stops linking `CUDA::cudart`
  globally — only backend targets link vendor runtimes. (Today cudart is
  linked into every first-party target including the scheduler; that is
  exactly the kind of quiet coupling this removes.)
- `csrc/` → `backends/cuda/`; `include/inferx/kernels/` → the neutral
  `include/inferx/ops/` (headers are already nearly vendor-free; the change
  is `cudaStream_t`→`Stream` and the include cleanup).
- Unguarded CUDA includes in `src/model/*.cc` / `src/engine/*.cc` (which
  today rely on CMake exclusion to stay out of host builds) disappear as a
  side effect: those files stop including vendor headers at all.

## 5. How hardware-specific code separates from core logic

Worked example — the pattern applied to the heaviest coupled class,
`Qwen2Model` (~55 memcpy sites, 6 pinned buffers, own stream/events, graph
capture):

| Today (vendor calls inline) | After (boundary calls) |
|---|---|
| `cudaStreamCreateWithFlags(&s, NonBlocking)` | `rt.CreateStream(dev)` |
| `cudaMemcpyAsync(dst, src, n, H2D, s)` | `rt.CopyAsync(dst, src, n, kH2D, s)` |
| `cudaHostAlloc(&p, n, Default)` | `rt.AllocPinnedHost(n)` |
| `cudaEventRecord(sampled_ready, s)` / `cudaEventSynchronize` | `rt.RecordEvent(e, s)` / `rt.SynchronizeEvent(e)` |
| `cudaStreamBeginCapture` … `cudaGraphInstantiate` … `cudaGraphLaunch` | `rt.BeginCapture(s)` … `rt.EndCaptureAndInstantiate(s)` … `rt.LaunchGraph(g, s)` |
| `kernels::RmsNorm(..., (cudaStream_t)stream)` | `ops::RmsNorm(..., stream)` |
| `gemm.LinearBF16(x, w, y, stream)` | unchanged (type behind `ops::Gemm` pimpl) |
| `INFERX_CUDA_RETURN_IF_ERROR(...)` | `INFERX_RETURN_IF_ERROR(rt.…)` — plain Status |

Every row is a mechanical substitution; none changes model logic, stream
ordering, or the "no synchronize in SubmitStep" design. The graph-capture
*strategy* (which shapes, when to capture, replay keys) stays in the model —
only the primitive operations move behind the runtime. The same table applies
to `GptOssModel`, `DeepseekV2Model`, `MlaAttentionLayer`, `MoeFfn`,
`WeightLoader` (its pinned-ring pipeline logic is vendor-neutral once
alloc/copy/event go through the runtime), the comm timing ring, and
`kv_autosize` (`GetMemInfo`).

Ownership rule after separation:

- **Core** owns: types (tensor/dtype/shape/device/stream handles), Status,
  the two boundary interfaces, allocator/arena/KV-pool logic.
- **Backends** own: everything that includes a vendor header — kernels,
  vendor-library wrappers, runtime implementation, error conversion,
  communicator transports.
- **Models/engine** own: orchestration only — what to run, in what order, on
  which stream, with which capture strategy. They compile with zero vendor
  headers and zero backend `#ifdef`s.
- **The `is_cuda()` naming sweep**: `Tensor::is_cuda()`, `CudaAvailable()`,
  `kCudaEnabled`, CLI strings like `--no-cuda-graphs` get neutral
  counterparts (`is_accelerator()`, `AcceleratorAvailable()`,
  `--no-device-graphs` with the old spellings as vLLM-compat aliases —
  consistent with the existing `vllm_compat.cc` accept-and-map approach).

## 6. Design decisions

| # | Decision | Alternatives rejected | Why |
|---|---|---|---|
| 1 | Two boundaries: op-granular compute (A) + fine-grained runtime services (B) | One giant backend interface; per-kernel virtual dispatch | Ops is where NPU vendor libraries plug in; runtime is where isomorphic APIs (cudart/HIP/aclrt) map 1:1. Different shapes, different contracts |
| 2 | Link-time backend selection, one device backend per binary | Runtime plugin registry; multi-backend fat binaries | Matches deployment; zero hot-path overhead; vastly less machinery. Op signatures don't preclude a registry later |
| 3 | Opaque `Stream`/`Event` value types in core; creation/lifetime on the runtime | Vendor types in headers (status quo); ref-counted stream objects | Zero-cost, fixes existing fragile transitive includes, proven by `Communicator`'s `void*` |
| 4 | ROCm via HIP shared-source port *inside* `backends/rocm` | HIP portability layer in core/csrc shared by both | Keeps core vendor-free; CUDA backend stays pristine nvcc; wavefront-size divergences handled where they belong |
| 5 | Ascend integrates at op granularity via aclnn/ATB, custom AscendC only as last resort | Reimplement all 58 kernels in AscendC | Vendor op libraries are the performant, supported path on NPU; kernel-parity is neither feasible nor necessary |
| 6 | Capabilities (`graph_capture`, `supported_compute`, …) gate features; engine already has no-graph and quantization-off paths | Assume feature parity; lowest-common-denominator API | Extends the proven `CommCapabilities` pattern; fail-fast at load, not mid-forward |
| 7 | `DeviceKind` per vendor + `IsAccelerator()`; kill `Cuda(0)` defaults | Generic `kGpu` kind; keep defaults | Kind must select the backend and error messages; defaults naming a vendor are the coupling being removed (aligned with `ParallelContext::device()`) |
| 8 | CPU becomes a real backend (runtime + reference ops where they exist) | Keep `#else` branches ad hoc | Makes the GPU-less CI story architectural; host-sim TP tests and the WeightLoader CPU path already want this |
| 9 | Error conversion is per-backend, core sees only `Status` | Central multi-vendor error enum | Each vendor's error space stays in its backend; the 199 macro sites migrate with the code they live in |
| 10 | `kernels::` → `ops::` rename with signature-preserving moves | Keep name; deep re-architecture of op API | The existing free-function surface is already the right API; the rename marks the contract without churning call sites beyond the stream type |
| 11 | Observability stays exporter-based per platform (DCGM → rocm/npu exporters) | In-process NVML/RSMI/DSMI integration | Zero in-process vendor telemetry exists today; keep it that way |

## 7. Testing strategy

- **Unit tier (no hardware)**: unchanged `ctest -L unit`; grows CPU-backend
  runtime tests (the interface contract runs against the CPU implementation).
- **Op conformance tier**: per-op differential tests, parameterized over the
  built backend, comparing against CPU reference implementations (which exist
  today for attention; added opportunistically elsewhere) and against
  recorded CUDA-reference outputs with per-op tolerances (§3.4.5). This tier
  is what makes "backend-agnostic" a tested property instead of a diagram.
- **Model tier**: the existing checkpoint validation gates
  (`docs/DSV2_VALIDATION.md`-style logit comparisons, greedy-token equality)
  run per backend; graph-on/graph-off equivalence where `graph_capture`.
- **A header-hygiene check**: a CI grep (or IWYU rule) asserting no vendor
  token appears in `include/inferx/` outside `include/inferx/backends/` —
  turning §1.2's leakage list into a ratchet that cannot regress.

## 8. Migration strategy and impact

Ordering principle: each phase is behavior-preserving on CUDA, lands green,
and is independently valuable. The CUDA build must never regress —
performance parity is an acceptance gate for every phase, and phases 1–3
change no kernel code at all.

Interaction with `tensor_parallel.md`: do **Phase 1 first** (opaque handles +
runtime interface), because the TP layer primitives
(`ColumnParallelLinear` etc.) should be authored against `Stream` and
`ops::` from day one — then the model-forward rewrite that TP adoption
already requires happens *once*, against the final surfaces. The TP work's
Qwen2 refactor and this design's Phase 3 are the same editing pass.

1. **Handles and runtime (mechanical, CUDA-only build unchanged)**
   Introduce `Stream`/`Event`, `DeviceRuntime` + CUDA and CPU
   implementations; reroute `DeviceBuffer`/`AllocatorFor`/`WeightLoader`
   internals; swap `cudaStream_t`→`Stream` in the 13 affected headers.
   Risk: low. Touches many signatures, changes no logic.
2. **Build restructure**
   `csrc/`→`backends/cuda/`, `kernels/`→`ops/` headers, `INFERX_BACKEND`
   switch, per-backend cmake files, remove global cudart linkage, add the
   header-hygiene ratchet. Risk: low; pure mechanics, big diff.
3. **Model/engine sweep (the substantive pass, shared with TP adoption)**
   Rewrite qwen2/gpt_oss/deepseek_v2/mla/moe_ffn/qwen_runner/kv_autosize per
   the §5 table; delete their vendor includes; neutral naming sweep.
   Gate: bit-identical outputs and step-time parity on CUDA at TP=1 and
   TP=2 (HostSim + NCCL), graphs on and off.
4. **CPU backend completion**
   Promote host-only build to a running configuration for the covered ops;
   op-conformance tier lands here. Value: hardware-free CI for everything
   above the kernels.
5. **ROCm bring-up** (first proof the boundary holds for a second GPU)
   Runtime, HIP kernel port + wavefront audit, hipBLASLt Gemm, RCCL,
   reference-attention fallback, then CK/aiter attention. Gate: model-tier
   validation on MI-series hardware.
6. **Ascend bring-up** (proof for non-GPU)
   aclrt runtime, aclnn/ATB ops for the dense path first (Qwen2), HCCL,
   `graph_capture=false` serving; MoE/MLA and quantized paths follow
   capability by capability.

### Impact summary

| Area | Impact |
|---|---|
| scheduler, server (43 files), api, tokenizer, observe, support | **Zero** (support: CLI alias strings only) |
| core | Small: `device.h` enum, new `stream.h`/`device_runtime.h`, reroute 2 `.cc` internals; `cuda_utils.h` relocates |
| ops headers (12) | Signature-only: stream type + include hygiene; call sites unchanged otherwise |
| `csrc/` 7.2k lines | Moves wholesale; zero kernel edits until Phase 5 |
| model (6 files) + engine (2 files) | The real work: ~55 memcpy sites, pinned buffers, events, graph calls → runtime calls; mechanical per the §5 table, merged with the TP refactor pass |
| comm | Enum values + `Stream` type; NCCL/HCCL/RCCL pattern already established |
| build | Restructured, behavior-preserving; host-only build strictly improves |
| tests/bench | `bench/cuda_timer.h` and kernel tests follow their backend; unit tier untouched |

## 9. Open questions

- **Attention on ROCm**: CK/aiter maturity for paged decode at our block
  sizes vs. living on reference kernels longer. Decide at Phase 5 with
  measurements; the ops API is unaffected.
- **Ascend KV layout negotiation**: whether `ops::PagedDecode` declares one
  preferred layout per backend or per (backend, model) — generalize
  `MlaAttentionLayer::LayoutFor` when ATB's actual constraints are known.
- **Device sampling portability**: the curand-based sampling path is
  CUDA-specific; CPU-side sampling after logit readback is the portable
  fallback. Whether ROCm gets hiprand parity in Phase 5 or falls back
  initially is a bring-up choice, capability-gated either way.
- **Unused vendored kernels** (CUTLASS, FlashMLA, hpc-ops): FlashMLA and
  hpc-ops are CUDA-only and PyTorch-/sm90-flavored; if adopted they enter as
  CUDA-backend internals behind existing ops. Whether to keep vendoring
  unreferenced submodules is a repo-hygiene question this design doesn't
  force.
- **`Range(1,2)` style caps and flags** (`--comm-backend nccl`): CLI grows
  `rccl`/`hccl` values when those backends land; whether `--comm-backend`
  should be inferred from `INFERX_BACKEND` entirely (one fewer knob) is a UX
  decision for Phase 5.
