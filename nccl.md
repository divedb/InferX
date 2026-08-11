# NCCL tensor-parallel design

## Status

Implemented through the optional NCCL backend, device-aware Qwen2 placement,
two-rank serving workers, CLI configuration, and concurrent graph dispatch.
Physical two-GPU correctness, failure injection, and performance validation are
pending suitable hardware.

The first production tensor-parallel target is one Linux process on one host
with exactly two CUDA GPUs. NCCL is the initial communication backend. The
model layer must continue to depend only on `comm::Communicator`, so another
backend such as MSCCL++ can be evaluated later without changing Qwen2 layer
code or its sharding rules.

M7 already validates the tensor-parallel arithmetic with `HostSimComm`: Q/K/V
and gate/up are output-sharded, O and down are input-sharded, and the two
row-parallel outputs are summed after every layer. This work replaces only the
runtime and transport. It does not change the mathematical partition.

## Goals

- Run Qwen2 tensor parallelism on two physical GPUs in one machine.
- Use one rank per GPU and NCCL in-place all-reduce on the model CUDA stream.
- Preserve TP=1 and deterministic `HostSimComm` tests.
- Keep backend types and headers out of model and server interfaces.
- Support CUDA graph capture after the uncaptured path is correct.
- Measure communication separately from model compute.
- Leave a clean integration point for MSCCL++ and other CUDA transports.

## Non-goals for the first implementation

- Multi-node execution or network bootstrap.
- More than one process.
- Elastic ranks, rank recovery, or communicator shrink/grow.
- Pipeline, expert, sequence, or context parallelism.
- Overlapping an all-reduce with the GEMM that produces its input.
- Backend selection per collective or automatic NCCL/MSCCL++ tuning.
- Production support for heterogeneous GPUs.

The interfaces must not preclude these, but the first implementation should
not pay their complexity cost.

## Current constraints

The existing abstraction is deliberately small:

```cpp
class Communicator {
 public:
  virtual int rank() const = 0;
  virtual int size() const = 0;
  virtual Status AllReduceSum(const TensorView&, void* stream) = 0;
};
```

Qwen2 issues two in-place BF16 sums per decoder layer, immediately after the O
projection and down projection. Every rank executes these collectives in the
same order.

Three pieces still assume the HostSim/single-GPU environment and must be fixed
before NCCL can be useful:

1. Many Qwen2 allocations and tensor views explicitly use `DeviceId::Cuda(0)`.
2. The server owns one model and has no rank workers or cross-rank failure
   propagation.
3. Graph capture currently rejects every communicator with `size() > 1`.

These are part of the NCCL work, not follow-up polish.

## Runtime topology

The first runtime uses one process with two long-lived worker threads:

```text
HTTP / scheduler coordinator
          |
          | immutable ForwardBatch for this step
          v
   +------+------+
   |             |
rank 0 worker  rank 1 worker
CUDA device 0 CUDA device 1
Qwen2 shard 0 Qwen2 shard 1
   |             |
   +--- NCCL ----+
```

Each worker:

1. calls `cudaSetDevice(local_device)` before CUDA or NCCL work;
2. owns one `Qwen2Model`, one CUDA stream, and one communicator rank;
3. receives the same logical `ForwardBatch` from the coordinator;
4. runs the rank-local forward step;
5. participates in collectives in exactly the same order as its peer;
6. returns status, timing, and rank-0 output to the coordinator.

Only the coordinator mutates scheduler state. Forward-batch host vectors are
read-only while workers upload them. Rank 0 supplies logits/sampled tokens to
the scheduler; rank 1's replicated logits need not be downloaded. Both ranks
must finish a step successfully before scheduler state is committed.

A thread-per-rank design is chosen over repeatedly switching devices on one
thread. CUDA current-device state is thread-local, it makes ownership obvious,
and the same worker boundary can later become one process per rank without
changing model or communicator APIs.

## Backend-neutral communicator contract

`Communicator` remains the model-facing interface. The first revision adds
device placement and explicit backend capabilities, but no NCCL types:

```cpp
enum class CommBackend { kSingleRank, kHostSim, kNccl, kMscclpp };

struct CommCapabilities {
  bool cuda_graph_capture = false;
  bool device_collectives = false;
};

class Communicator {
 public:
  virtual ~Communicator() = default;
  virtual int rank() const = 0;
  virtual int size() const = 0;
  virtual DeviceId device() const = 0;
  virtual CommBackend backend() const = 0;
  virtual CommCapabilities capabilities() const = 0;
  virtual Status AllReduceSum(const TensorView&, void* stream) = 0;
};
```

`void* stream` remains intentionally opaque at this boundary. NCCL and the
prospective MSCCL++ backend both consume CUDA work on a CUDA stream, while the
host backend ignores it. Model code must not include `nccl.h` or MSCCL++
headers.

The collective contract is:

- in-place sum, with the result available on every rank;
- asynchronous with respect to the host when a CUDA stream is supplied;
- ordered after prior work and before subsequent work on that stream;
- identical dtype, count, collective order, and communicator on every rank;
- tensor residence must equal `communicator.device()`;
- zero-element tensors succeed without calling the backend;
- supported production dtype initially is BF16; host-test dtypes stay as-is;
- launch errors are returned immediately when the backend reports them;
- asynchronous errors are checked at step synchronization boundaries.

Capabilities replace checks such as `comm->size() > 1` for graph capture. This
allows NCCL to opt in without pretending HostSim's host staging is capturable.

## NCCL backend

### Construction

The public factory accepts backend-neutral configuration:

```cpp
struct NcclCommConfig {
  int rank;
  int world_size;
  int local_device;
  NcclUniqueIdBytes unique_id;
};

StatusOr<std::unique_ptr<Communicator>> CreateNcclCommunicator(
    const NcclCommConfig&);
```

`NcclUniqueIdBytes` is a fixed byte container declared without including
`nccl.h`. An NCCL-specific bootstrap helper creates it once on the coordinator,
then copies it into both worker configurations. Internally the factory converts
it to `ncclUniqueId` and calls `ncclCommInitRank` for the worker's fixed rank and
device. NCCL requires every rank to map to a distinct CUDA device; construction
validates the rank range, device range, world size, and unique-ID size.

The implementation is an optional `inferx_comm_nccl` target. Configure-time
detection uses `find_path(nccl.h)` and `find_library(nccl)`. A build without
NCCL retains TP=1 and HostSim. Requesting the NCCL backend from such a build
returns a clear `Unimplemented` status.

For the initial single-process target, `ncclCommInitAll` is possible, but it is
not used: `ncclGetUniqueId` plus `ncclCommInitRank` mirrors the eventual
multi-process bootstrap and tests the lifecycle that future deployment needs.

### Collective dispatch

`AllReduceSum` validates the tensor and maps InferX dtype to `ncclDataType_t`.
For BF16 it calls:

```cpp
ncclAllReduce(data, data, count, ncclBfloat16, ncclSum, comm, cuda_stream);
```

There is no device or stream synchronization inside the method. Synchronizing
there would destroy overlap and graph capture. The model already places the
projection, collective, and following residual operation on one stream, which
provides the required dependency.

### Ownership and destruction

An `NcclComm` owns one `ncclComm_t` and its fixed `DeviceId`. It does not own
the model stream. Destruction happens on the rank worker after outstanding work
has been synchronized. Normal teardown calls `ncclCommFinalize` where supported
and then `ncclCommDestroy`. Failure teardown calls `ncclCommAbort` so one failed
rank cannot leave the peer waiting forever.

All NCCL calls go through one status-conversion helper that records the NCCL
error name and message. CUDA errors and NCCL errors remain distinguishable in
the returned text.

### Failure propagation

Collective mismatches generally manifest as a hang, so prevention is more
important than recovery:

- the coordinator dispatches a monotonically increasing step ID to both ranks;
- both workers receive the same batch shape and collective sequence;
- no rank may return early after entering model execution;
- a local validation failure is reported before either worker starts the step;
- the first runtime failure sets shared cancellation state and aborts both NCCL
  communicators;
- the scheduler commits no tokens or KV ownership for a failed step;
- the engine becomes unhealthy and rejects later requests rather than trying to
  reuse uncertain rank/KV state.

The worker join path must have a bounded diagnostic timeout. A timeout is not
normal control flow; it dumps rank, step ID, last collective ordinal, CUDA
device, and NCCL async status before aborting the communicator.

## Device-aware model changes

All Qwen2 CUDA placement derives from `comm->device()`:

- weight uploads and concatenated weights;
- activation, quantization, sampling, and index buffers;
- KV block pools;
- `TensorView::Create` device metadata;
- CUDA stream and event creation;
- FlashInfer workspaces and plans.

No model allocation may contain a literal `DeviceId::Cuda(0)`. Public load
overloads that create `SingleRankComm` retain device 0 by default, preserving
current callers.

The checkpoint mapping may be shared read-only across rank loaders, but the
first implementation may load each rank independently for simplicity. GPU
uploads occur concurrently only after correctness is established; serialized
upload avoids unnecessary host-memory and PCIe contention during bring-up.

Each rank owns a local KV pool with the already validated local KV-head count.
Physical block decisions remain identical because the coordinator supplies the
same logical block table. No KV data is communicated between ranks.

## Engine integration

Add a tensor-parallel section to `EngineConfig`:

```cpp
struct TensorParallelConfig {
  int world_size = 1;
  std::vector<int> devices;  // default [0] or [0, 1] for TP=2
  CommBackend backend = CommBackend::kSingleRank;
};
```

CLI shape for the first release:

```text
--tensor-parallel-size 2
--device-ids 0,1
--comm-backend nccl
```

Rules:

- TP=1 defaults to `SingleRankComm` and one model, with no worker overhead.
- NCCL initially requires TP=2 and exactly two distinct local devices.
- model head, KV-head, and intermediate dimensions must divide TP size before
  allocating GPU memory;
- every selected GPU must have enough free memory for its shard, local KV pool,
  activations, NCCL scratch, and graph capture;
- mixed GPU models are rejected initially;
- HTTP and scheduling remain on the coordinator; no rank opens a server port.

The engine's existing synchronous `Step` contract is preserved externally.
Internally it becomes a two-worker barrier. Rank 0's result is returned only
after rank 1 also succeeds.

## CUDA graphs

Graph capture is enabled only after uncaptured two-GPU inference passes. NCCL
collectives are recorded on each rank's model stream; both rank workers begin
capture for the same graph shape, execute the same collective sequence, end
capture, and instantiate independently.

Requirements:

- `capabilities().cuda_graph_capture` is true on every rank;
- communicator and all buffer addresses outlive every graph executable;
- graph capture and replay are entered concurrently on both rank workers;
- no NCCL initialization, allocation, host synchronization, or error polling
  occurs inside capture;
- graph keys include TP size and backend in addition to batch shape;
- a failed capture destroys partial graphs on both ranks and falls back to the
  uncaptured path unless strict graph mode was requested.

Graph replay must be diffed against uncaptured NCCL execution before it becomes
the default.

## MSCCL++ extension point

MSCCL++ should be a separate implementation of the same `Communicator`, not a
conditional branch inside `NcclComm`. It can report the same device-collective
and graph capabilities while owning its bootstrap, channels, memory
registrations, proxy service, and kernels.

This boundary is deliberate. MSCCL++ exposes lower-level GPU-driven channels,
one-sided transfers, zero-copy paths, and customizable collective kernels. Its
potential value is application-specific optimization of the relatively small
all-reduces in LLM decode, but adopting those abstractions in model code would
couple every layer to one experimental transport.

The first comparison should keep semantics identical:

- in-place BF16 sum;
- same model stream ordering;
- same two GPUs, batch shapes, clocks, and graph setting;
- same numerical tolerance and generated tokens;
- backend selected once when the engine is created.

Only after a backend-level all-reduce benchmark and end-to-end decode benchmark
show a repeatable gain should MSCCL++ become more than an optional experiment.

## Validation plan

### Phase 1: communicator

- Build with and without NCCL installed.
- Reject invalid rank/device/world-size configurations.
- Two-GPU in-place BF16 all-reduce with known values.
- Reuse the communicator for many collectives of changing valid sizes.
- Run on non-default CUDA streams and prove stream ordering without a hidden
  synchronization.
- Inject a rank failure and verify bounded abort/teardown.
- Run `nccl-tests` and record topology plus measured bus bandwidth.

### Phase 2: device-aware Qwen2

- Assert every allocation belongs to its rank's selected device.
- Run the real checkpoint at TP=1 and NCCL TP=2.
- Compare full prefill, paged prefill, and incremental decode logits.
- Require identical greedy tokens and retain the existing numerical tolerance.
- Verify each rank's weight and KV allocation is locally sharded.

### Phase 3: serving

- Start one HTTP server backed by two rank workers.
- Exercise concurrent requests, prefix cache, preemption, and cancellation.
- Verify worker errors fail the engine without deadlock or scheduler commit.
- Compare graph-disabled and graph-enabled generation.

### Phase 4: performance

Record:

- `nvidia-smi topo -m` and P2P capability;
- GPU model, VRAM, driver, CUDA, NCCL, PCIe/NVLink state, and clocks;
- raw all-reduce latency/bandwidth for the actual Qwen2 message sizes;
- per-layer collective time;
- prefill throughput, decode latency, and output throughput for TP=1 vs TP=2;
- graph capture benefit;
- compute/communication fraction and scaling efficiency.

TP=2 is successful only if it is correct, stable under serving load, and its
performance result is explained by measured compute and interconnect costs. A
slower TP=2 result can still validate the implementation, but it must not be
presented as a scaling win.

## Implementation sequence

1. Add backend/device/capability metadata to `Communicator` and preserve all
   SingleRank/HostSim tests.
2. Add optional NCCL discovery, status conversion, unique-ID bootstrap, and
   `NcclComm` with focused two-GPU tests.
3. Remove device-0 literals from Qwen2 and KV allocation; validate TP=2 model
   execution directly.
4. Add engine rank workers, shared step dispatch, failure propagation, and CLI.
5. Enable and differential-test multi-rank CUDA graph capture.
6. Benchmark the rented two-GPU host and record its topology and versions.
7. Evaluate MSCCL++ only against the established NCCL correctness and
   performance baseline.

## Acceptance criteria

- A default build and TP=1 behavior are unchanged when NCCL is absent.
- A two-GPU NCCL build serves the real Qwen2 checkpoint with correct prefill
  and incremental decode.
- There are no implicit host copies or device synchronizations in production
  collectives.
- Model and scheduler sources contain no NCCL or MSCCL++ API types.
- Rank failure cannot leave the server indefinitely blocked in a collective.
- Captured and uncaptured NCCL paths generate the same tokens.
- Documentation includes the exact tested cloud topology and benchmark data.

## References

- NVIDIA NCCL, “Creating a Communicator”:
  https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/communicators.html
- NVIDIA NCCL, “CUDA Stream Semantics”:
  https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/streams.html
- NVIDIA NCCL, GPU troubleshooting and P2P topology:
  https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/troubleshooting/gpu_troubleshooting.html
- MSCCL++ repository and architecture overview:
  https://github.com/microsoft/mscclpp
