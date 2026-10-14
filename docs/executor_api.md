# Executor API after model loading

Status: proposed redesign, not an implemented API. Written 2026-10-14.

Revision: expanded execution coverage, including concrete worker handoff,
TP/PP, speculative and asynchronous execution, multimodal/pooling operations,
state management, and operational controls. All capability claims below describe
the proposed contract; none asserts that InferX currently implements it.

The executor accepts an immutable, already scheduled batch, executes it across
the loaded model's worker group, and returns explicit execution results. The
scheduler owns request history and committed progress. A cache manager owns
logical cache allocation. Rank-local runners own tensor preparation, attention
metadata, model invocation, logits, sampling, and device completion.

This is a fresh contract. Existing executor, scheduler-output, and runner APIs
are not compatibility constraints. C++ examples use InferX's `Status`,
`StatusOr`, `Tensor`, and `Model` vocabulary; new types below are proposed types,
not claims about available headers. Model construction, checkpoint parsing,
weight loading, tokenizer work, and HTTP serving are outside this document.
Explicit adapter installation and quiescent runtime maintenance are included;
they never implicitly replace base model weights during execution.

## 1. What the reference implementations establish

The following observations come from upstream `main` sources read on
2026-10-14. These links are mutable, not pinned release guarantees. The proposed
InferX contract in subsequent sections is our design, not either project's API.

| Source inspected | Observed boundary | Design consequence |
| --- | --- | --- |
| [vLLM Executor](https://github.com/vllm-project/vllm/blob/main/vllm/v1/executor/abstract.py) | Worker coordination; KV configuration and warmup; separate execution and sampling calls; blocking/nonblocking results and health/shutdown. | Separate lifecycle from batch execution, and separate local computation from rank coordination. |
| [vLLM SchedulerOutput](https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/sched/output.py) | New requests, cached-request deltas, token allocations, finished requests, speculative tokens, and connector metadata. | The scheduler chooses work. Persistent request mirroring is an optimization requiring its own consistency protocol. |
| [vLLM GPUModelRunner](https://github.com/vllm-project/vllm/blob/main/vllm/v1/worker/gpu_model_runner.py) | Updates batch state, constructs execution inputs, runs the model, selects hidden rows for the head, and retains intermediate state for sampling. Its async result retains tensors through the host copy. | Make selected rows and buffer lifetime explicit; do not equate a kernel launch with completion. |
| [vLLM ModelRunnerOutput](https://github.com/vllm-project/vllm/blob/main/vllm/v1/outputs.py) | Request identities, variable-length sampled-token lists, logprobs, and optional auxiliary/connector results. | Return identities and progress explicitly; sequence count and output count are different dimensions. |
| [SGLang ForwardBatch](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/model_executor/forward_batch_info.py) | Documents `ScheduleBatch -> ForwardBatch`; carries token tensors, sequence lengths, request-pool indices, output cache locations, and execution mode. | Use a distinct device batch inside the runner; include both attention read geometry and cache write locations. |
| [SGLang TpModelWorker](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/managers/tp_worker.py) | Constructs a forward batch, calls the runner, samples generation outputs, can defer sampling, and skips ordinary sampling for verification. | Keep forward and sampling separable internally, with a single ordinary-generation entry point externally. |
| [SGLang ModelRunner](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/model_executor/model_runner.py) | Initializes attention execution metadata and selects graph replay or a live execution path according to batch eligibility. | Graph selection and backend metadata belong below the scheduler boundary. |

The important distinction is scope: vLLM's distributed executor is not equivalent
to SGLang's `model_executor` package. The latter contains rank-local execution
machinery. InferX should name and expose those responsibilities separately.

Additional sources inspected for the expanded contract:

| Source | Relevant observation |
| --- | --- |
| [MultiprocExecutor](https://github.com/vllm-project/vllm/blob/main/vllm/v1/executor/multiproc_executor.py) | Supports PP and coordinates worker messages/results; topology and output-rank selection are executor concerns. |
| [SamplingMetadata](https://github.com/vllm-project/vllm/blob/main/vllm/v1/sample/metadata.py), [sampler](https://github.com/vllm-project/vllm/blob/main/vllm/v1/sample/sampler.py), [penalties](https://github.com/vllm-project/vllm/blob/main/vllm/v1/sample/ops/penalties.py) | Carry separate prompt/output history, generators, constraints, processors, and selected score requests. Scoring representation is configurable. |
| [SpecDecodeMetadata](https://github.com/vllm-project/vllm/blob/main/vllm/v1/spec_decode/metadata.py), [rejection sampler](https://github.com/vllm-project/vllm/blob/main/vllm/v1/sample/rejection_sampler.py) | Verification maps draft, target, and bonus rows explicitly and can use draft probabilities. |
| [KV cache specifications](https://github.com/vllm-project/vllm/blob/main/vllm/v1/kv_cache_interface.py) | Full, sliding-window, latent-attention, and recurrent-state geometries need different allocation/retention rules. |
| [KV connector](https://github.com/vllm-project/vllm/blob/main/vllm/distributed/kv_transfer/kv_connector/v1/base.py) | Separates scheduler metadata from worker transfers; asynchronous saves must retain source blocks through completion, including work driven by empty steps. |
| [PoolingMetadata](https://github.com/vllm-project/vllm/blob/main/vllm/v1/pool/metadata.py) | Pooling has task-specific parameters, completion tracking, and state for chunked outputs. |
| [GPU input batch](https://github.com/vllm-project/vllm/blob/main/vllm/v1/worker/gpu_input_batch.py) | Persistent worker-side batching is a separate concern from a scheduler's authoritative request state. |
| [Parallel configuration](https://github.com/vllm-project/vllm/blob/main/vllm/config/parallel.py), [speculative configuration](https://github.com/vllm-project/vllm/blob/main/vllm/config/speculative.py) | Feature composition has configuration-specific constraints; support cannot be inferred from independent booleans. |

The coverage matrix in section 11 is the design target for the majority of
ordinary vLLM execution workflows. It is not a measured percentage of vLLM
features, a promise of identical numerics, or a claim that every feature pair
works in every upstream backend. Source observations above and InferX decisions
below are deliberately distinguished.

## 2. Ownership and supported baseline

| Component | Owns | Does not decide |
| --- | --- | --- |
| Engine/scheduler | Admission, ordering, token budgets, canonical token history, cancellation, stop conditions, committed lengths | Kernel selection or device pointers |
| Cache manager | Page identities, reservations, prefix lookup, sharing, copy-on-write, eviction, commit/abort of reservations | Which model layers or kernels run |
| Executor | Loaded worker group, lifecycle, accepted submissions, resource retention, collective ordering, result aggregation | Admission, preemption, prefix policy |
| Rank-local runner | Streams, scratch, tensor packing, attention plans, graphs, model forward, head, sampler | Canonical request history or logical page allocation |
| Loaded model | Immutable weights, architecture, model/state requirements, forward and head computation | Request scheduling or cache policy |

The core causal batch covers token inputs, chunked prefill, mixed prefill/decode,
sampling, and selected-row scoring. The complete design also defines PP,
speculation, multimodal/encoder tasks, pooling, adapters, cache transfers, hybrid
state, dependent decode, and collective idle steps in sections 12–18. These are
typed operations with explicit capability requirements, not undefined future
hooks. A minimal implementation may expose only a subset while preserving the
same contracts. Section 20 separates implementation milestones from API scope.

Each executor owns one replica's workers. Independent DP replicas use separate
executors; coupled DP/EP deployments additionally use a domain coordinator
(section 12). CPU, CUDA, other accelerators, eager/graph modes, and quantized
weights/cache are backend capabilities, not different public scheduler APIs.
No enum or configuration field alone constitutes support.

## 3. Public surface

```cpp
namespace inferx::execution {

class Executor {
 public:
  // Consumes an already loaded group. Never opens a checkpoint.
  static StatusOr<std::unique_ptr<Executor>> Attach(
      LoadedWorkerGroup workers, ExecutorOptions options);
  ~Executor();

  const ExecutorInfo& info() const;
  ExecutorState state() const;
  ArtifactStore& artifacts(); // Bounded artifact access; see section 15.

  StatusOr<MemoryReport> Profile(const ProfilePlan& plan);
  StatusOr<ReadyInfo> Initialize(const RuntimePlan& plan);

  StatusOr<AdmissionReport> CanExecute(const WorkDescriptor& work) const;

  // Validation is useful to callers but Submit always validates again.
  Status Validate(const ExecutionBatch& batch) const;
  StatusOr<ExecutionTicket> Submit(ExecutionBatch batch);
  StatusOr<BatchResult> Execute(ExecutionBatch batch); // Submit + Wait

  // Typed task envelope; the causal overload above is a convenience wrapper.
  Status Validate(const ExecutionRequest& request) const;
  StatusOr<ExecutionTicket> Submit(ExecutionRequest request);

  StatusOr<ForwardTicket> Forward(ForwardRequest request);
  StatusOr<ExecutionTicket> Resolve(ForwardTicket& ticket,
                                    ResolutionPlan plan);
  Status Discard(ForwardTicket& ticket);

  StatusOr<ProposalTicket> Propose(DraftRequest request);
  StatusOr<SettlementReceipt> Settle(const Settlement& decision);
  StatusOr<ControlTicket> Control(ControlCommand command);

  StatusOr<HealthReport> CheckHealth(Deadline deadline) const;
  Status Drain(Deadline deadline);
  Status Shutdown(ShutdownMode mode, Deadline deadline);
};

class ExecutionTicket { // Move-only; belongs to one accepted submission.
 public:
  SubmissionId id() const;
  StatusOr<std::optional<BatchResult>> Poll();
  StatusOr<BatchResult> Wait(Deadline deadline);
  StatusOr<DeviceTokenRef> ExportDependency(SequenceKey sequence,
                                           uint32_t output_index);
};

} // namespace inferx::execution
```

`Deadline` is an absolute monotonic time or infinity. `ExecutorOptions` contains
bounded submission capacity, worker timeout, and diagnostic settings; it contains
no scheduler policy. `SubmissionId` is unique within an executor incarnation.
Public batch/control mutations are serialized by one engine thread. Ticket
`Poll`/`Wait` may run on another thread, with only one consumer per ticket.
`CanExecute` validates a metadata-only `WorkDescriptor` against the loaded
model, backend, topology, operation, state kind, dtype, shape, graph policy,
adapter set, and requested feature combinations. It returns supported limits,
required resources, and typed incompatibility reasons. `Validate` also checks
actual payloads, references, and leases; `Submit` atomically reserves admission
credits and rechecks mutable state. Neither query guarantees later admission.

Health and state queries are thread-safe. Internal workers may execute in
parallel. A second mutating caller receives `FailedPrecondition` rather than
depending on unspecified interleaving.

Unless explicitly stated otherwise, every ticket is move-only; its result is
retrieved once, a caller deadline is nonterminal, and dropping it never releases
live device resources. `ForwardTicket` exposes `id()` and readiness/error polling
without consuming its retained forward state; only resolve/discard consumes it.
`ProposalTicket` returns `ProposalArtifact`; `ControlTicket` returns
`ControlResult`. `ExportDependency` must run before the execution result is
consumed, does not consume it, and pins the referenced device output independently
of ticket lifetime. Non-generating operations reject dependency export.

Administrative query types have the following minimum contents:

| Type | Required contents |
| --- | --- |
| `WorkDescriptor` | Task kind, model/adapter revisions, topology, input/output shapes, state layout IDs, requested processors/features, graph policy, byte estimates |
| `AdmissionReport` | Supported/rejected decision, effective limits, per-rank resource estimates, incompatibility codes and affected fields; no resource reservation |
| `ReadyInfo` | Execution profile, runtime/topology epochs, actual capacities, state-pool descriptors, lease authority, graph coverage |
| `HealthReport` | Group state, per-worker liveness/last-progress time, latched errors, outstanding submissions/transfers, quiescence blockers |
| `ResolutionBudget` | Allowed output kinds/rows, maximum score/draft/output counts, retained-logit and processor bytes, retention deadline |
| `SettlementReceipt` | Producer/sequence, applied history revision, kept output/state boundary, decision identity, descendant disposition |

### Loaded-worker handoff

`LoadedWorkerGroup` is a move-only owner of workers with loaded models, initialized
device runtimes, and established communicators. An in-process worker owns a
`unique_ptr<Model>`; a process worker owns a transport endpoint to its resident
model. Attachment verifies the model/vocabulary fingerprint, weight revision,
rank topology, shard compatibility, and readiness on every worker.

```cpp
struct ModelIdentity {
  std::string architecture;
  std::string config_fingerprint;
  std::string weights_revision;
  std::string vocabulary_fingerprint;
};

struct RankPlacement {
  uint32_t worker_id;
  uint32_t tp_rank;
  uint32_t pp_stage;
  uint32_t layer_begin; // Inclusive; model-module mapping supplements this.
  uint32_t layer_end;   // Exclusive.
};

struct WorkerTopology {
  uint32_t tp_size;
  uint32_t pp_size;
  std::vector<RankPlacement> ranks;
  std::vector<CommunicationGroup> groups;
  uint32_t output_worker_id; // In final PP stage; not necessarily rank zero.
};

class LoadedWorkerGroup {
 public:
  LoadedWorkerGroup(LoadedWorkerGroup&&) noexcept;
  LoadedWorkerGroup& operator=(LoadedWorkerGroup&&) noexcept;
  LoadedWorkerGroup(const LoadedWorkerGroup&) = delete;
  LoadedWorkerGroup& operator=(const LoadedWorkerGroup&) = delete;
  ~LoadedWorkerGroup();

 private:
  friend class ModelLoader;
  friend class Executor;
  LoadedWorkerGroup(ModelIdentity identity, WorkerTopology topology,
                    std::vector<std::unique_ptr<WorkerHandle>> workers,
                    std::vector<LoadedAuxiliaryGroup> auxiliaries);
  ModelIdentity identity_;
  WorkerTopology topology_;
  std::vector<std::unique_ptr<WorkerHandle>> workers_;
  std::vector<LoadedAuxiliaryGroup> auxiliaries_;
};
```

`WorkerHandle` owns either a local loaded worker or an authenticated transport
and lifecycle lease for a remote worker. It is not a raw `Model*`. Teardown of
externally managed workers releases model/session ownership; it does not kill
unrelated processes. `CommunicationGroup` specifies an ID, kind (TP, PP, EP,
context, or domain control), ordered member IDs, backend, and incarnation.
EP/context membership is explicit; its size is not assumed to multiply TP/PP
sizes. Plain TP×PP has exactly `tp_size * pp_size` workers with unique coordinates.

Workers already have resident immutable weight shards, initialized runtimes,
and ready communicators. Serving cache, bounded staging/workspaces, and captured
graphs are created by `Initialize`. Auxiliary groups contain separately loaded
draft or encoder models with their own identities, topology, and resource
budgets. Only declared auxiliary dependencies can be invoked.

Attach verifies full, nonoverlapping decoder layer coverage across PP stages,
matching TP shard/module layouts within stages, embedding/head ownership,
activation interface compatibility, and tied-weight handling at the endpoints.
Models with noncontiguous stage modules publish a module-placement manifest;
the layer range alone does not prove a valid partition.

Transfer of ownership is explicit even if attachment fails: the consumed group
is shut down safely. Loading an auxiliary/draft model or replacing weights is
not a side effect of any executor method.

`ExecutorInfo` contains an incarnation ID, model fingerprint and immutable model
facts, global vocabulary/context limits, worker topology/output owner, per-rank
state-layout requirements, supported attention layouts/dtypes, supported output
and sampling operations, graph support, and implementation limits. A capability
must hold across all participating ranks.

Capabilities are versioned operation profiles with compatibility predicates,
not an unrestricted Cartesian product of feature flags. For example, support
for PP and for speculation separately does not authorize their combination.
`Initialize` freezes an `ExecutionProfileId` identifying enabled operations,
sampling semantics, resource limits, and allowed combinations. Unsupported
combinations produce a reason naming the conflicting features before launch.

### Lifecycle

```text
Attach -> Attached -> Profile (repeatable) -> Initialize -> Ready
                                              failure -> Attached or Failed
Ready -> Draining -> Ready
Ready -> Quiescing -> Quiesced -> Sleeping -> Waking -> Quiesced -> Ready
Attached / Ready / Failed -> Shutdown -> Closed
Any unrecoverable worker/device error -> Failed
```

`Profile` runs bounded synthetic shapes using isolated scratch/cache state after
weights are resident. Its report records, per rank, resident weight bytes,
non-cache persistent bytes, peak activation/workspace bytes for the tested
shapes, device free bytes, and graph budget estimates. Measurements identify the
profile plan; they are estimates for planning, not a guarantee about untested
shapes. Profile requires `Attached` and never touches serving request state.

`RuntimePlan` specifies `max_sequences`, `max_input_tokens`, `max_logit_rows`,
`max_context_tokens`, `max_inflight_batches`, per-rank workspace budgets, cache
group page counts/layouts, and graph policy (`Disabled`, `Prefer`, `Require`)
with capture shapes. Enabled capacities must be positive and fit backend integer
ranges; disabled task-specific capacities are zero.
Distinct token, sequence, and head-row limits are mandatory.
The expanded plan also bounds encoder tokens/items, retained encoder bytes,
recurrent snapshots, draft tokens/probability bytes, PP activation bytes and
microbatches, unresolved forward tickets, dependency depth, transfer bytes,
adapter residency, and result/artifact bytes. A disabled capability has zero
budget; inference of unlimited capacity from a missing field is prohibited.

`Initialize` allocates rank-local KV storage, staging/scratch/output pools, and
optional graph buffers; warms representative execution paths; then publishes
`ReadyInfo` containing actual limits, cache-pool descriptors, and graph coverage.
Warmup uses private dummy pages and never consumes live logical reservations.
Graph persistent memory and maximum in-flight buffer use are budgeted before
declaring ready. Partial initialization is cleaned up; retry is allowed only
if every rank returns to a confirmed `Attached` state. Otherwise enter `Failed`.

For TP, logical page capacity is bounded by the least capable participating
rank; page bytes may differ by shard. The cache manager receives descriptors
and an opaque lease authority, not a mutable rank-zero pool pointer. Storage
addresses remain stable within a runtime epoch. Sleep/wake with reallocation
changes that epoch and invalidates affected handles/graphs (section 18).
Capacity resizing requires a new executor.

## 4. Scheduler-to-executor batch

The public contract is a complete description of this step. It does not require
`AddRequest`, `UpdateRequest`, `FinishRequest`, or implicit "last sampled token"
state. Full here means the current inputs and cache mappings, not resending the
entire prompt on every decode step. Sampling history is included only when an
enabled processor needs it.

```cpp
using TokenId = int32_t;
using Position = int64_t;
using PageId = uint32_t;
using CacheGroupId = uint32_t;

struct SequenceKey {
  uint64_t request_id;
  uint64_t branch_id;        // Independent n-sample or beam branch.
  uint64_t generation;       // Changes on restart/recompute or ID reuse.
};

struct SequenceSlice {
  SequenceKey key;
  uint64_t history_revision; // Engine's canonical history version.
  Position prefix_tokens;   // Valid cached tokens BEFORE this step.
  uint32_t input_begin;      // Offset in flattened token_ids.
  uint32_t input_count;      // Strictly positive.
  AdapterRef adapter;        // Explicit base-model sentinel or pinned revision.
};

struct PageRef {
  PageId id;
  uint64_t allocation_epoch; // Rejects a freed/reused page identity.
};

struct CacheTable {
  uint32_t sequence_index;
  CacheGroupId group;
  std::vector<PageRef> pages; // In logical sequence order.
};

struct ExecutionBatch {
  uint64_t engine_step;      // Trace/correlation; not a retry token.
  std::vector<SequenceSlice> sequences;
  std::vector<TokenId> token_ids;
  std::vector<CacheTable> cache_tables;
  CacheLease cache_lease;
  OutputPlan outputs;
  DecoderExtensions extensions; // Typed bindings defined in sections 13–17.
};
```

`CacheLease` is a move-only, cache-manager-issued reservation retaining every
referenced page and immutable prefix validity metadata. It records the executor
incarnation/pool generation, read regions, exclusive write regions, and any
cache-preparation completion fences. It cannot be fabricated from raw page IDs.
The executor checks tables against it. On submission rejection its reference is
released; on acceptance the executor retains it through actual device completion.
The cache manager separately retains the reservation until commit or abort.

For sequence `i`, let `P = prefix_tokens`, `Q = input_count`, and
`O = input_begin`. Input row `O+j` has causal position `P+j`, for `0 <= j < Q`.
Its attention reads the valid prefix and preceding/current tokens from this
chunk. For ordinary generation the materialized length is `P+Q`; speculative
execution separately bounds the valid accepted prefix (section 14).
The executor derives
positions; scalar positions cannot disagree with cache positions. Models needing
multidimensional or remapped positions use the typed position binding in section 15.

Prefill, cached-prefix extension, and decode share this geometry. `Q=1` alone
does not mean a token should be sampled. Sampling is an explicit output request.
The runner may classify or reorder sequences for optimized kernels, but public
results retain input indices and keys.

### Required validation

Before accepting work, check all of the following:

* Ready state, queue capacity, and negotiated batch limits.
* Sequence keys are unique in the batch. Slices cover `token_ids` exactly in
  sequence order, without gaps or overlap. All counts, sums, and position
  arithmetic are checked for overflow; concrete token-embedding rows lie in
  `[0, vocab_size)`. Explicit embedding/dependency rows follow their own schema.
* `P >= 0`, `Q > 0`, and `P+Q` fits both model and configured context limits.
* There is exactly one table per required cache group per sequence. For the
  baseline full-attention group with page size `B`, it has exactly
  `ceil((P+Q)/B)` pages. Unused entries of the last page are not valid tokens.
* The lease proves `[0,P)` is valid for this sequence/history and grants writes
  to `[P,P+Q)`. Page epochs, pool generation, and model identity match.
  Hybrid groups use section 16's per-group intervals/checkpoints, and validated
  dependencies use section 13's provisional-prefix exception.
* Writers never alias another writer or a shared read region. A shared partial
  tail must already have been copied into an exclusive page. Copy fences may
  remain pending at acceptance but are waited on before model execution.
* Output rows belong to the stated sequences, all operations are supported,
  and sampling/scoring parameters and masks are internally consistent.

An empty batch with empty tables/output plan is a valid completed no-op and does
not launch kernels unless it is wrapped in an explicit collective-domain epoch.
`MaintenanceStep` represents connector-only or required idle participation
(sections 12 and 17); empty user input never silently creates dummy tokens.

`Validate` changes no state and reserves nothing. A successful call is not an
admission guarantee: capacity or health can change before `Submit`.

## 5. Selected logits and sampling

```cpp
enum class SamplingMode { Greedy, Stochastic };
enum class ScoreSpace { RawLogits, RawLogprobs,
                        ProcessedLogits, ProcessedLogprobs };

struct SamplingPolicy {
  SamplingMode mode = SamplingMode::Greedy;
  float temperature = 1.0f;
  uint32_t top_k = 0;
  float top_p = 1.0f;
  float min_p = 0.0f;
  float repetition_penalty = 1.0f;
  float frequency_penalty = 0.0f;
  float presence_penalty = 0.0f;
  std::vector<TokenBias> logit_bias;
  uint32_t min_generated_tokens = 0;
  std::vector<TokenId> suppressed_stop_ids;
  std::vector<std::vector<TokenId>> banned_sequences;
  std::vector<ProcessorRef> processors;
};

struct SamplingHistory {
  std::vector<TokenCount> prompt_counts;
  std::vector<TokenCount> generated_counts;
  uint64_t generated_length;
  std::vector<TokenId> ordered_suffix;
};

struct RngKey {
  uint64_t seed;
  SequenceKey sequence;
  uint64_t generated_index;
  uint32_t purpose;
};

struct ScoreRequest {
  ScoreSpace space = ScoreSpace::RawLogprobs;
  uint32_t top_k = 0;
  bool include_selected_ranks = false;
};

struct LogitRow {
  uint32_t sequence_index;
  uint32_t input_offset; // Relative to that sequence's current chunk.
};

struct SampleSpec {
  uint32_t logit_row;    // Index in OutputPlan::rows.
  SamplingPolicy policy;
  RngKey rng;
  SamplingHistory history;
  std::optional<AllowedTokenSet> allowed_tokens;
  std::optional<ScoreRequest> score_request; // Absent means no score work.
};

struct ScoreSpec {
  uint32_t logit_row;
  std::vector<TokenId> token_ids;
  ScoreRequest score_request;
  std::optional<ScoreProcessing> processing; // Policy/history/mask, if processed.
};

struct OutputPlan {
  std::vector<LogitRow> rows;
  std::vector<SampleSpec> samples;
  std::vector<ScoreSpec> scores;
  std::optional<VerificationPlan> verification;
};
```

`TokenBias` is `(TokenId, finite float bias)` and `TokenCount` is
`(TokenId, positive uint64 count)`. `ScoreProcessing` bundles a policy, history,
and allowed-token mask without a sampling/RNG request. Selected IDs for a sample
are the sampled ID; selected IDs for a score-only row are `ScoreSpec::token_ids`.

The runner computes the transformer for all input tokens, then projects only
the unique requested hidden rows. The same logit row can support both sampling
and scoring without an additional head computation. Empty `rows` skips the LM
head and sampler; the step is cache-only unless it also requests hidden artifacts.
Raw scores/exports retain unmodified logits before any in-place processor runs.

In ordinary generation, at most one `SampleSpec` exists per sequence and it
references that sequence's final input row. The scheduler requests it only when
all known inputs have been consumed. Intermediate prompt chunks have no sample.
Independent samples use different branch keys; beam search uses candidate scores
and cache forks (section 11). Verification uses `verification`, which is mutually
exclusive with ordinary samples for the same sequence. Its head-row and progress
rules are defined in section 14.

`SamplingPolicy` explicitly carries greedy/stochastic mode, temperature, top-k,
top-p, min-p, repetition/presence/frequency penalties, sparse logit biases, and
processor extensions for minimum generation length and banned token sequences.
`ScoreRequest` independently selects score semantics and top-score count.
The minimal supported policy is greedy with neutral processors. Any unsupported non-neutral field causes rejection; it must never
silently fall back to argmax.

Policy validation requires finite numeric values, positive repetition penalty,
positive stochastic temperature, `0 <= top_k <= vocab_size`, `0 < top_p <= 1`,
`0 <= min_p <= 1`, and a top-logprob count within the negotiated limit. Sparse
biases and allowed-token IDs must be unique and in vocabulary range. Score
targets may repeat; their results preserve caller order. Output-plan rows are
unique by `(sequence_index, input_offset)` and every row must have a consumer.
Deferred forwards reserve their consumers through `ResolutionBudget` instead.
IDs in biases, masks, stop IDs, and banned sequences must be in range; banned
sequences must be nonempty. The ordered suffix must cover the longest history
required by enabled processors. A generated-length counter alone is insufficient
to enforce multi-token bans.

The default processing profile applies whitelist/banned/minimum-length masks,
additive bias, repetition and frequency/presence penalties, temperature, min-p,
top-k, top-p, and renormalization. Registered processors have fixed positions
in this versioned profile. Repetition penalty divides positive logits and multiplies negative logits by the positive penalty;
repetition considers prompt and generated tokens, while frequency/presence use
generated-token counts. `SamplingHistory` supplies separate `prompt_counts`,
`generated_counts`, `generated_length`, and an ordered suffix for sequence-based
processors. These describe canonical history through the sampling position,
including known current inputs, never an unverified future draft suffix. Counts
are unique by token ID and positive. History may be omitted only when no active
processor requires it. Banned token sequences and minimum-length EOS suppression
are converted into masks using this history before truncation. Registered
processor IDs carry a version, declared history requirements, ordering, and
graph/speculation compatibility; arbitrary callbacks are not sent to workers.
`allowed_tokens` is an explicit whitelist: absent means all tokens, present and
empty is invalid. Grammar state stays with the engine, which computes the mask.

Min-p removes probabilities below its fraction of the largest probability after
temperature. Top-k retains at most k tokens (`0` disables it); top-p then retains
the smallest descending prefix whose cumulative mass, renormalized over current
survivors, reaches p. Ranking ties use token ID.
Greedy mode rejects non-neutral temperature/top-k/top-p/min-p settings and
chooses the lowest token ID among maximum processed logits. NaNs, positive
infinities, or no remaining finite logit fail the batch; negative infinity is
allowed for masked/impossible tokens. Unsupported parameters are validated first.

`RngKey` contains a request seed, generation, absolute generated-token index,
and draw-purpose tag. Sampling uses a specified counter-based algorithm so batch
order and interleaving do not change random draws. The executor has no global
advancing request RNG. The engine advances its generated-token index only after
committing an output. Reproducible draws do not promise numerically identical
probabilities across hardware or precision choices.

`ScoreRequest` defines `ScoreSpace::{RawLogits, RawLogprobs, ProcessedLogits,
ProcessedLogprobs}`, `top_k`, and whether selected-token
ranks are required. Raw logprobs are log-softmax before processors; processed
values use the declared sampling pipeline, including truncation. Greedy mode
can return raw or processed model scores; these are not the log probability of
a point-mass argmax decision. `ScoreSpec` defaults to raw logprobs; processed
score-only rows must provide an explicit policy/history binding. Results echo
their score space, and masked tokens have negative-infinite logprobs. This
avoids conflating generation probabilities with prompt/model likelihoods.

A logit at absolute position `p` scores the token at `p+1`. For chunk-boundary prompt scoring, score the first
token of the next chunk using the previous chunk's final row. The first prompt
token has no model likelihood without an explicit predecessor such as BOS.
Score-only rows need not be final rows. Full-vocabulary or hidden-state export is
an explicit, budgeted artifact operation (section 15), never an automatic host
copy. Sampling semantics carry a profile version; compatibility with a selected
vLLM release must be tested rather than inferred from matching parameter names.

## 6. Result and completion contract

```cpp
struct SequenceProgress {
  SequenceKey key;
  uint64_t history_revision;
  Position prefix_before;
  Position executed_after;     // Includes provisionally evaluated draft inputs.
  Position materialized_after;
};

struct SampleResult {
  uint32_t sample_index; // Index in the submitted samples array.
  SequenceKey key;
  TokenId token;
  std::optional<TokenScores> scores;
};

struct ScoreResult {
  uint32_t score_index;
  TokenScores scores; // Selected values preserve ScoreSpec token order.
};

struct BatchResult {
  SubmissionId submission;
  uint64_t engine_step;
  std::vector<SequenceProgress> progress; // Input sequence order.
  std::vector<SampleResult> samples;       // SampleSpec order.
  std::vector<ScoreResult> scores;         // ScoreSpec order.
  std::vector<VerificationResult> verification;
  std::vector<ArtifactResult> artifacts;  // Encoder/pooling/hidden outputs.
  std::vector<StateCheckpoint> checkpoints;
  std::vector<TransferProgress> transfers;
  ExecutionStats stats;
};
```

Successful compute completion means every worker has finished all reads, writes,
collectives, and result copies associated with the batch. Host result vectors
own their memory independently of runner scratch. No returned span refers to
reusable graph buffers. Scalar outputs are host-ready. Artifact handles and
device dependencies have their own ownership/readiness contract in sections
13 and 15; successful host completion does not release referenced artifacts.
`TokenScores` carries score space, selected IDs/values, optional 1-based ranks,
and top entries sorted by descending value with token-ID tie breaking.
Maintenance results are snapshots: a reported pending transfer is not completed
DMA. Its independent lease remains held until the terminal receipt (section 17).

`materialized_after` reports valid computed inputs, **not** committed request
progress and **not** generated-token count. A newly sampled token has not yet
been fed through the model and has no KV entry. The engine validates the key and
history revision, commits the computed prefix, appends an allowed output token,
evaluates stop conditions, and publishes client-visible output. The executor
does not produce an authoritative finish reason.

`Submit` consumes its by-value batch and returns after validation, resource
reservation, and dispatch acceptance; it does not wait for GPU completion.
A synchronous rejection launches no model work and leaves committed KV intact.
Transport/device errors after acceptance are reported through the ticket.
Failures return a non-OK `Status`, not partially successful sequence results.

`Poll` returns an empty optional while pending, a result on successful terminal
completion, or terminal error. `Wait` blocks until completion or its deadline.
A wait timeout is a local observation and leaves the ticket live; it neither
cancels execution nor permits page reuse. Successful or failed terminal retrieval
consumes the ticket result; further retrieval returns `FailedPrecondition`.
For `Poll`, `DeadlineExceeded` is never used to mean merely pending.
Worker execution timeouts are terminal `Unavailable` errors with timeout details,
so `Wait` callers can distinguish them from their own nonterminal deadline.
`Execute` waits indefinitely and has exactly the same terminal result semantics.

Dropping a ticket discards interest in its result. The executor still owns the
submission and lease until work is terminal and safe. Ticket destruction never
frees live buffers. Completed but unconsumed results count against bounded
submission capacity; consuming or dropping the ticket releases that capacity.

There is no automatic retry and no network-style exactly-once promise. A returned
`SubmissionId` identifies accepted work; a repeated `engine_step` is not deduped.
The caller must establish resource safety and abort the old reservation before
retrying with a new lease; a promptly reported fatal error is insufficient.

## 7. KV storage and commit rules

For a full-attention group, the rank-local runner derives each write slot as:

```text
logical_position = prefix_tokens + input_offset
page = table[logical_position / tokens_per_page]
slot = physical_page_base(page.id) + logical_position % tokens_per_page
```

The scheduler never supplies a device address. Each rank maps the same logical
page identities to its own storage, head shard, layout, and stride. Group
descriptors associate layers with compatible cache geometry. Sliding-window,
latent-attention, and recurrent layouts must define their own typed translation
and validity rules before admission.

The cache-manager contract required by the executor is:

1. Reserve and pin a known-valid prefix plus exclusive append capacity.
2. Resolve shared partial tails with copy-on-write before append; attach copy
   completion fences to the lease.
3. Retain all referenced storage until execution reaches safe completion.
4. On success, let the engine commit a valid prefix length no greater than the
   reported materialized length. Publish only committed prefix entries.
5. On discard or failure, abort the reservation after safe completion. Written
   but uncommitted tail bytes remain invisible and are overwritten before reuse.

Existing valid prefix bytes must never be overwritten by ordinary append.
Consequently a per-batch sampling failure can invalidate its newly written suffix
without destroying the previously committed prefix. This is a visibility rule,
not a promise that GPU writes can be physically rolled back.

In the ordinary host-driven path, only one in-flight batch may touch a sequence,
and lease hazard checks reject incompatible physical-page access across different sequence
keys. Shared immutable prefix reads are allowed. A next decode batch may be
submitted only after the engine commits its predecessor. Overlapping independent
batches and CPU preparation are allowed up to the negotiated bound. Section 13
defines the additional ordering, provisional state, and settlement required for
dependent device-side decode; enabling async submission alone does not enable it.

If all prompt tokens hit a KV prefix cache, KV alone does not recover the last
hidden state/logits. To sample the first output, schedule at least the final prompt
token again, using the valid prefix ending immediately before it and an exclusive
tail page. A cache that explicitly stores reusable logits could supply a future
separate path; a zero-token batch cannot sample from KV alone.

## 8. Rank-local execution interface

These are executor-private interfaces, not scheduler methods. The executor
dispatches one accepted step to every participating worker in a common order.

```cpp
class ModelRunner {
 public:
  StatusOr<PreparedBatch> Prepare(const ExecutionBatch& batch,
                                BufferSlot slot);
  StatusOr<ForwardResult> Forward(PreparedBatch& batch);
  StatusOr<LocalResult> ResolveOutputs(PreparedBatch& batch,
                                     ForwardResult forward);
};

struct DeviceBatch {
  Tensor token_ids;        // int32 [T]
  Tensor positions;        // int64 [T], or checked backend conversion
  Tensor query_offsets;    // int32 [S+1], ragged offsets
  Tensor prefix_lengths;   // int32 [S], checked conversion
  Tensor sequence_lengths; // int32 [S], prefix + chunk
  Tensor logit_indices;    // int32 [R], flattened input row indices
  std::vector<GroupAttentionInputs> attention;
};
```

`S` is sequence count, `T` the flattened input-token count, and `R` the unique
logit-row count. `GroupAttentionInputs` holds device block tables, per-row write
slots, and backend-specific attention metadata for one cache group.
`PreparedBatch` owns/retains the buffer slot, immutable host snapshot, device
batch, cache views, scratch lease, and completion events. `ForwardResult` owns
or retains hidden/logit buffers until output resolution completes.

Execution order:

1. Acquire an in-flight buffer slot and stage inputs on the copy stream.
2. Make the compute stream wait on staging and cache-preparation fences.
3. Build backend attention metadata, including ragged causal masks and writes.
4. Run the loaded model's forward computation with rank-local cache views,
   scratch, stream, and communicator in its execution context.
5. Select requested hidden rows and compute the head; skip when `R=0`.
6. Perform required vocabulary communication and resolve samples/scores.
7. Copy compact results to host, join worker completion, and complete the ticket.

The loaded model's forward/head split is useful, but this proposal does not
require the existing `ForwardContext`, runner constructors, or pool APIs to
survive unchanged. A model never receives an engine `Request*`.

### Tensor parallelism

All ranks agree on batch identity, sequence order, row selection, per-stage graph routes,
and collective ordering before launching model collectives. Preflight allocation
and capability failures are aggregated before launch. Rank-local page addresses
are allowed to differ; logical mappings are not.

The designated output owner resolves each sample exactly once. All ranks still
participate in required vocabulary/other collectives. The baseline may gather
selected logits to that owner; optimized distributed sampling must preserve the
same probability and token-ID semantics. Because subsequent batch inputs are
explicit, persistent per-rank token-history mirroring is unnecessary.

An asynchronous rank failure aborts the worker group and fails its pending work.
No rank is allowed to return a successful batch while another remains in a
collective. Health checks use control-plane liveness and backend error state,
not an unscheduled collective that could collide with model operations.

### Graphs and buffer reuse

Graph eligibility includes token/sequence/row capacities, attention layout,
execution shape, output operation, dtype, and topology. The runner owns graph
keys and padding. Padded rows use private safe slots and never write live request
pages. Result indices exclude padding.

With `Prefer`, an uncovered shape uses eager execution; with `Require`, validation
rejects it before launch. Replay inputs live at stable addresses and are copied
into a graph slot only after its previous consumer has completed. Either provide
separate buffers/graph instances for overlapping work or serialize access to a
shared instance. Graph replay does not relax lease or stream ordering rules.

## 9. Cancellation, errors, and shutdown

Request cancellation is an engine decision. Before submission, simply do not
submit. After submission, mark the sequence cancelled, suppress its output, and
wait for terminal execution before aborting its reservation. The baseline has no
`CancelRequest` executor method because kernels and collective groups are not
independently cancellable per sequence.

| Condition | Result | May the executor continue? |
| --- | --- | --- |
| Invalid shape, token, row, policy | `InvalidArgument` before acceptance | Yes |
| Unsupported operation/layout | `Unimplemented` before acceptance | Yes |
| Wrong state, stale lease, overlapping write | `FailedPrecondition` before acceptance | Yes |
| Submission/buffer capacity exhausted | `ResourceExhausted` before acceptance | Yes |
| Caller wait deadline expires | `DeadlineExceeded`; submission remains live | Yes; retain resources |
| Sampling distribution invalid after forward | Terminal batch error; discard its suffix | Yes after all ranks are quiescent |
| Unexpected device/collective/worker failure | Terminal error, executor enters `Failed` | No; recreate group |

Error diagnostics include submission/engine-step IDs, failing stage and rank,
and whether the executor is failed. A fatal error may be reported promptly before
hardware quiescence; the lease authority must quarantine affected pages until
worker teardown establishes safety. A ticket error alone is never permission
to bypass the lease authority and reuse physical storage.

`Drain` stops accepting submissions while it waits for all already accepted work
to reach safe completion. It does not consume ticket results. Success reopens
admission in `Ready`; a timeout leaves admission closed until a subsequent
successful drain or shutdown. Internally this requires a `Draining` state.
If completion requires a caller action not yet queued (resolve, discard, or
settlement), `Drain` returns `FailedPrecondition` with blocker IDs instead of
waiting indefinitely on the same calling thread. Cleanup operations remain
allowed while draining. An async `Quiesce` control can instead remain pending
while the engine supplies those actions.

`ShutdownMode::Drain` closes admission, drains, and releases resources;
`ShutdownMode::Abort` closes admission, fails pending work, aborts communicators,
and tears down workers before releasing affected storage. Shutdown is idempotent.
A timeout leaves the executor closing and resources retained; callers may retry
shutdown. Internally `Closing` distinguishes this state from `Closed`.
The destructor performs abort-and-join cleanup and may block; explicit shutdown
is required when the application needs a deadline. Dropped tickets do not keep
the executor's model workers alive after successful shutdown.
Drain shutdown reports unresolved logical-work blockers rather than implicitly
sampling or committing them; abort shutdown discards them. Host-owned retrieved
results survive shutdown. Device artifact/dependency handles are revoked after
their readers are safe; later access returns a stale-incarnation error. No
external wrapper can keep the executor's device allocations alive indefinitely.

`ExecutionStats` records actual input-token/sequence/logit-row counts, eager or
graph route and padding counts, queue delay, and optionally measured device/copy
times. Profiling timers are optional to avoid adding hot-path synchronization.
Memory reports distinguish persistent cache, workspace, graphs, and retained
outputs. No per-token success logging is required.

## 10. Concrete examples

### Chunked prefill and first decode

For prompt `[a,b,c,d,e]`, page size four, chunk size three:

| Step | Valid prefix P | Inputs Q | Requested sample row | Successful result |
| --- | --- | --- | --- | --- |
| 1 | 0 | `[a,b,c]` | None | Materialized length 3, no generated token |
| 2 | 3 | `[d,e]` | Local row 1 (`e`) | Materialized length 5, sampled `x` |
| 3 | 5 | `[x]` | Local row 0 (`x`) | Materialized length 6, sampled `y` |

Step 1 needs one page, steps 2/3 need two. After step 2 the canonical history has
six tokens but the valid KV prefix has five. Treating the sampled token as
already computed would skip its model evaluation in step 3.

### Mixed batch

Sequence A has `P=3,Q=2` and inputs `[d,e]`; sequence B has `P=20,Q=1` and input
`[z]`. Pack `token_ids=[d,e,z]`, `query_offsets=[0,2,3]`,
`positions=[3,4,20]`, `sequence_lengths=[5,21]`. If both generate, logit rows
are `{A,1}` and `{B,0}`, flattened rows `[1,2]`. The attention backend may use
separate optimized paths internally without changing the result contract.

### Engine call sequence

```cpp
// Loader has already produced resident models and communicators.
auto executor = TRY(Executor::Attach(std::move(loaded_workers), options));
auto memory = TRY(executor->Profile(profile_plan));
auto ready = TRY(executor->Initialize(ChooseRuntimePlan(memory)));
auto cache = MakeCacheManager(ready.cache_pools, ready.lease_authority);

// Schematic engine/cache calls, not additional Executor methods.
auto reservation = TRY(cache.Reserve(scheduled_work));
auto batch = BuildExecutionBatch(scheduled_work, reservation.MakeLease());
auto ticket = TRY(executor->Submit(std::move(batch)));
// CPU can prepare unrelated work here.
auto outcome = ticket.Wait(Deadline::Infinite());
if (!outcome.ok()) {
  cache.AbortWhenSafe(reservation);
  return outcome.status();
}
for (const auto& progress : outcome->progress) {
  if (!engine.StillCurrent(progress.key, progress.history_revision)) {
    cache.AbortSequenceWhenSafe(reservation, progress.key);
    continue; // Cancelled/replaced; publish nothing for this sequence.
  }
  engine.CommitSequenceProgressAndOutputs(reservation, progress, *outcome);
}
// Unrelated live sequences still commit when another has been cancelled.
```

`TRY` is pseudocode for checked `StatusOr` propagation. Production engine code
must also abort reservations on submission rejection and use deadline-aware
ticket handling without treating a wait timeout as a terminal failure.

## 11. Use-case coverage and observable outcomes

Each row is a proposed supported workflow when its named capability profile is
enabled. The minimal implementation is not the definition of the complete API.
Every workflow must either execute with these semantics or reject before model
launch with a specific unsupported-combination reason.

| ID / workflow | Caller action and execution path | Observable result / essential edge case |
| --- | --- | --- |
| U01 Offline generation | Reserve cache, submit whole prompt, then feed sampled IDs back as decode inputs. | Same contracts as serving; no server or tokenizer dependency. |
| U02 Continuous batching | Mix new/chunked prompts and decode slices; reorder the next batch freely. | Results map to keys and operation indices, never transient batch slots. |
| U03 Chunked prefill | Submit chunks without sampling until the final prompt row. | No dummy output on intermediate chunks; cancellation leaves only committed prefixes reusable. |
| U04 Prefix reuse | Lease a matched prefix, append into exclusive tail pages. | Full cache hit still requires last-row recomputation for generation unless a compatible logits artifact exists. |
| U05 Prefix cache with prompt scores | Request score rows for every required next-token pair. | KV-only hits cannot reconstruct skipped scores; recompute with scratch/COW storage or reuse an explicitly cached score artifact. |
| U06 Greedy/stochastic sampling | Supply policy, canonical history, RNG key, masks, and score request per sequence. | Mixed policies in one batch; unsupported processors rejected, all-masked distribution fails explicitly. |
| U07 Structured output | Submit ready grammar masks, or `Forward` then `Resolve` after grammar computation. | Mask revision bound to sequence/history; missing mask is not unconstrained sampling. |
| U08 Multiple outputs / beam search | Fork cache leases into branch keys; independently sample or request top candidate scores. | Engine owns branch selection, cumulative scores, length penalties, and termination; executor never silently allocates beams. |
| U09 Preemption / recomputation | Stop scheduling a sequence, settle in-flight work, release or export state; resume from a valid prefix with a new generation. | Old results rejected; resume cannot trust freed page IDs or stale recurrent snapshots. |
| U10 TP and multi-node workers | Submit one logical step to a loaded rank group. | One output owner; consistent collective order, page identity and error propagation. |
| U11 PP and TP×PP | Partition scheduled work into microbatches and transport stage activations. | Full result after all stages; bounded activation credits prevent pipeline deadlock. |
| U12 Independent DP / coupled EP | Independent executors, or coordinator-issued collective-domain epochs. | Empty local batches still participate when the domain requires collectives. |
| U13 Context parallelism | Bind a supported context partition and attention-state placement. | Global positions/causality preserved; unsupported cache layout or graph combination rejected. |
| U14 Host/device overlap | Submit unrelated work asynchronously; optionally bind a typed previous-token dependency. | Device-ready is distinct from host-ready and engine-committed; cancellation invalidates descendants. |
| U15 Speculative generation | Propose or supply a chain, verify target rows, settle an accepted output prefix. | Variable token counts and separate executed/valid lengths; rejected draft KV never published. |
| U16 Image/audio/video and prompt embeddings | Encode typed preprocessed inputs or bind retained encoder/embedding artifacts. | Placeholder mapping, modality position tensors, encoder identity and artifact lifetime validated. |
| U17 Encoder-decoder generation | Encode source once, then submit decoder tokens plus cross-attention state. | Encoder/source length is independent of decoder position and self-attention prefix. |
| U18 Embeddings / classification / reward scoring | Submit model-supported `PoolingBatch` with output transform. | Returns vectors/scalars/token tensors without inventing generated tokens or calling an LM sampler. |
| U19 Hybrid full/window/recurrent models | Bind per-group pages, retained intervals, and recurrent snapshots. | State commits atomically across groups; recurrent rollback is explicit. |
| U20 LoRA and quantized execution | Pin adapter revision per sequence; negotiate loaded kernel/cache quantization. | Cache identity includes adapter and numerical representation; unavailable kernels fail admission. |
| U21 Disaggregated prefill/decode | Export committed prefix, import into reserved destination state, then decode. | Transfer receipt and readiness fences required; producer output-token handoff prevents duplicated first token. |
| U22 Host/storage offload | Export/import state through typed transfer commands. | Source retained during DMA; partial transfer never becomes a valid prefix. |
| U23 Connector-only / idle step | Submit `MaintenanceStep` with no decoder tokens. | Processes transfer completion or required collectives; is not mistaken for a no-op. |
| U24 Graph replay / fallback | Submit within profile, select matching captured shape or eager fallback. | Padding never touches user pages; `Require` rejects uncovered combinations. |
| U25 Sleep / wake / cache reset | Quiesce, execute typed control, inspect invalidation receipt, resume. | Runtime epochs and graph recapture prevent stale handles after reallocation. |
| U26 Diagnostics / shutdown | Read health, start bounded trace, drain or abort. | Pending work and resource safety remain observable; telemetry cannot reorder collectives. |

For U08, a branch receives its own `(request_id, branch_id, generation)`, RNG
stream, and write lease. Shared prompt pages remain immutable. Beam candidates
are score results, not yet committed tokens; the scheduler chooses children and
forks their cache state. `top_k` candidate count and returned-score bytes are
bounded. This supports engine-driven beam workflows without a hidden executor
beam-search loop or a requirement to execute all candidates.

Zero `max_new_tokens` is an engine policy: choose cache-only, score-only, or no
work. An empty prompt needs an engine-supplied BOS or a model-supported embedding
input. EOS, stop strings, output limits, and streaming text offsets remain engine
decisions. No execution method inserts BOS/EOS or detokenizes implicitly.

## 12. Parallel execution and pipeline protocol

### Pipeline stage interface

```cpp
struct MicrobatchKey {
  SubmissionId submission;
  uint32_t index;
};

struct ActivationPacket {
  MicrobatchKey microbatch;
  uint64_t topology_epoch;
  ActivationSchemaId schema;
  std::vector<OwnedTensorRef> tensors; // Hidden, residual, model-specific fields.
  DeviceFence ready;
};

struct StageResult {
  std::optional<ActivationPacket> next_stage;
  std::optional<LocalResult> final_outputs;
  DeviceFence local_work_done;
};

class PipelineRunner {
 public:
  StatusOr<StageResult> RunStage(
      PreparedMicrobatch& batch,
      std::optional<ActivationPacket> incoming);
};
```

Stage zero receives tokens/embeddings; middle stages require an activation packet;
the final stage performs final normalization, requested head/pool operations, and
output resolution. Stage schema includes tensor names, shapes, dtype, TP
partitioning/replication, and residual conventions; one `[T,H]` tensor is not a
universal PP boundary. Cross-stage resharding must be an explicitly supported
transport operation. `OwnedTensorRef` pins allocation and source runtime until
the transfer and all readers complete. A local fence is represented remotely by
a transport completion token, never a serialized CUDA event pointer.

The executor makes a `PipelinePlan` containing microbatch membership, stage
routes, activation credits, deterministic send/receive order, and output-row
remapping. It partitions whole sequence slices first. Splitting one sequence's
chunk further requires explicit causal dependencies and prefix visibility; it
cannot be treated as independent microbatches. Each stage's TP group executes
the same microbatch in the same order; distinct stages can overlap different
microbatches. A stage need only hold cache groups for its local layers.

Example: TP=2, PP=3, with microbatches A and B:

```text
time slot       stage 0 (TP group)   stage 1 (TP group)   stage 2 (TP group)
0               A                   —                   —
1               B                   A                   —
2               next work           B                   A + output head
3               ...                 ...                 B + output head
```

Execution dependencies, rather than literal equal-duration slots, determine
real scheduling. One microbatch alone remains correct but underutilizes stages.
The engine supplies admitted work; the executor chooses its physical execution
schedule without changing token budgets or admitting new requests.

Required guarantees:

* Before launch, reserve receive buffers and bounded downstream capacity, or
  establish a credit protocol with an acyclic wait order. Never wait for a
  receive buffer whose owner is blocked on the same send.
* Tag messages with incarnation, submission, microbatch, stage, and topology
  epoch. A late packet cannot satisfy another microbatch's receive.
* The last-stage output owner samples once. TP ranks elsewhere need no sampled
  history for host-driven execution. Device-dependent decode forwards sampled
  IDs and validity metadata to the first stage through explicit dependencies.
* Aggregate progress across all relevant stages and restore original output
  order. A microbatch's completion alone does not complete the public batch.
* Failure after any stage writes invalidates the batch's provisional suffix on
  every stage. Abort outstanding sends/receives and collectives; quarantine
  storage until the whole affected group is safe.

### DP, EP, context partitioning, and idle work

Independent dense DP replicas do not share an execution-order requirement.
For MoE EP spanning replicas, or any backend requiring synchronized DP epochs,
a `CollectiveDomainCoordinator` assigns `DomainEpoch {domain_id, topology_epoch,
step, participants, global_shape, collective_schedule}` before local submission.
Each participating executor receives a real task or explicit idle task for that
epoch. A nonparticipant cannot enter its collectives. The coordinator reserves
all participants' admission credits before releasing the epoch to launch.

Expert routing, all-to-all, token permutations, capacity handling, and restoring
token order live in the runner/model. Idle ranks execute the backend-required
empty or padded participation using scratch state. They produce no user tokens.
Control changes affecting a shared EP domain must quiesce the entire domain.

`ContextPartition` identifies the ordered workers, owned query/key position
ranges, replication rules, and reduction scheme supported by the attention
backend. Global positions and causal masks remain authoritative. Each cache
descriptor states physical placement; contiguous page ownership cannot be
assumed. Prefill and decode context partitioning are separate negotiated modes.

Uniform TP×PP is the common profile. Heterogeneous stage widths, elastic EP
repartitioning, and arbitrary topology changes are explicitly outside the first
complete profile; use a new loaded group instead of mutating communicators under
accepted work. This is a deployment limitation, not a silent execution fallback.

## 13. Deferred sampling and dependent asynchronous decode

### Two-phase forward and output resolution

```cpp
struct ForwardRequest {
  ExecutionBatch batch; // Rows fixed; samples/scores/verification unresolved.
  ResolutionBudget budget;
};

struct ResolutionPlan {
  std::vector<SampleSpec> samples;
  std::vector<ScoreSpec> scores;
  std::optional<VerificationPlan> verification;
};
```

`Forward` accepts work under the same admission/lease rules as `Submit` and
returns a move-only `ForwardTicket`. The request fixes input geometry, adapter,
head rows, and maximum output/processor workspace. The ticket retains logits,
cache leases, and intermediate state. `Resolve` may be called before forward
completion; it queues a consumer on the forward fence. It may choose only
declared rows and resources within the reserved budget.

`Resolve(ticket, plan)` validates without consuming the ticket on synchronous
rejection, so a stale grammar mask can be corrected. Acceptance consumes the
forward ticket and returns an ordinary execution ticket. No second resolution
is allowed. `Discard` consumes the ticket, suppresses outputs, and releases
resources only after safe completion; it does not roll back committed prefixes.
Dropping the forward ticket has the same effect. A bounded retention deadline
can discard abandoned logits and returns a terminal expiration error on later
use. It never frees a live device buffer.

Grammar masks carry sequence key, history revision, grammar-state revision, and
target generation index. The engine computes the mask for exactly that prefix.
The resolver checks identity and shape, including speculative per-prefix masks;
it never interprets a missing mask as permission to generate freely. If a grammar
cannot be evaluated for speculative paths, that combination is rejected or the
engine chooses ordinary decoding explicitly.

### Device-token dependency

```cpp
struct DeviceTokenRef {
  SubmissionId producer;
  SequenceKey sequence;
  uint32_t output_index;
  uint64_t expected_history_revision;
};

struct DependentInput {
  uint32_t input_row;
  DeviceTokenRef token;
  ProvisionalStateRef parent_state;
};

struct Settlement {
  SubmissionId producer;
  SequenceKey sequence;
  uint64_t expected_history_revision;
  uint64_t new_history_revision;
  uint32_t keep_output_tokens;
  Position keep_materialized_tokens;
  CacheCommitReceipt cache_receipt;
};
```

`ExecutionTicket::ExportDependency(sequence, output_index)` returns an executor-
validated, reference-counted dependency handle without waiting for host output.
The struct above describes its identity, not a forgeable authority. Binding it
in `DecoderExtensions::dependencies` replaces the indicated placeholder input
row; that placeholder is never embedded or counted as a real token. `Submit`
validates the producer, acyclic ordering, capacity, and state lease immediately;
the actual token and validity are checked after the producer's device fence.

The state manager grants a child reservation referencing the parent's provisional
state. This is the only exception to the committed-prefix lease rule. Child
reads are ordered after parent writes and child writes target distinct suffix
slots. Recurrent state uses separate versioned snapshots. Overlap must not
overwrite anything required to discard the parent or any earlier committed state.

`Settle` records an engine decision after the engine/cache manager has validated
the output and issued a matching `CacheCommitReceipt`. The receipt proves the
same producer, sequence/revision, kept output prefix, and retained state lengths;
the executor does not decide request history. A decision is idempotent for an
identical payload; conflicting duplicates fail. The executor retains a bounded
settlement journal until all descendants/reference handles are released.

Children may compute provisionally, but their outputs cannot become externally
committed until ancestor settlements validate their exact input/state prefix.
Cancellation, stop truncation, invalid grammar, or rejection invalidates affected
descendants. Descendant reservations are aborted after device completion. If a
descendant shares a public batch with unrelated work, the conservative contract
fails that whole batch; retry unrelated work from its old committed prefixes.
This avoids an ambiguous partially successful result.

This path has a maximum dependency depth and separately reserved continuation
credits so parents can always resolve and settle when the ordinary queue is
full. Resolution, settlement, failure propagation, and cleanup have priority
over fresh work. The common optimized profile supports one-token greedy or
stochastic decode with device-compatible processors; variable-length speculative
dependencies require a separate compatibility profile. Waiting for a ticket
alone neither commits it nor advances RNG/history revisions.

## 14. Speculative proposal, verification, and commit

```cpp
struct DraftRequest {
  SequenceKey sequence;
  uint64_t history_revision;
  std::optional<AuxiliaryModelId> proposer; // Absent for history-only methods.
  ProposalMethod method; // Model, n-gram, auxiliary-head, or feature-based.
  uint32_t max_draft_tokens;
  DraftInputs inputs;    // Tokens/history and/or declared hidden artifacts.
  StateLease draft_state;
  SamplingPolicy policy;
  RngKey rng;
};

struct DraftChain {
  uint32_t sequence_index;
  std::vector<TokenId> tokens;
  ProposalDistribution distribution;
};

struct VerificationPlan {
  VerificationAlgorithm algorithm; // GreedyMatch or ExactRejection.
  std::vector<DraftChain> chains;
  std::vector<VerificationRows> rows;
  std::vector<SampleSpec> target_policies;
};

struct VerificationResult {
  SequenceKey sequence;
  uint32_t accepted_drafts;
  std::vector<TokenId> output_tokens; // Accepted prefix + recovery/bonus.
  std::vector<TokenScores> scores;
  Position executed_after;
  Position materialized_after;
};
```

`Propose` uses only attached auxiliaries or a registered history-based proposer;
`ProposalTicket` has the same poll/wait/deadline ownership conventions as an
execution ticket and returns a proposal artifact stamped with target vocabulary,
draft revision, history revision, and probability semantics. Feature-based
methods declare which target layers/norm stages they consume. A draft model may
use a different topology/cache layout, but identical token semantics are required
for this profile. Vocabulary translation is not implicit.

The engine reserves target capacity and submits an ordinary causal batch whose
sequence input is `[anchor, d0, ..., d(K-1)]`. The anchor is the next known,
uncomputed canonical token. `VerificationRows` maps local rows `0..K-1` to draft
predictions and row `K` to the bonus distribution; arrays can be ragged across
sequences. K=0 is ordinary next-token generation. Verification still runs the
head for K+1 rows, not one row per sequence.

`GreedyMatch` accepts the longest prefix agreeing with target argmax and emits
the target token at the first mismatch, or a bonus token after full acceptance.
`ExactRejection` requires the actual normalized proposal distribution q for
each proposed prefix, or a declared deterministic point mass. For target p,
accept a proposed token d with probability `min(1, p(d)/q(d))`; on rejection,
draw from normalized `max(p-q,0)` and stop that chain. After all drafts are
accepted, draw a bonus token from the next target distribution. Unknown q is not
permission to claim exact stochastic verification. Full-q artifacts may stay
on device; their lifetime and bytes are charged to the proposal budget.

Target penalties, grammar masks, and stochastic transforms are applied for each
hypothetical prefix, not once using the history after all drafts. Draft and target
distributions record their respective processing versions. Acceptance/recovery/
bonus randomness uses disjoint purpose tags; distributional correctness does not
require bitwise matching to an ordinary sampling stream. Nonfinite distributions
or impossible normalization fail the verification result explicitly.

If the starting valid prefix is P and A drafts are accepted, then:

```text
evaluated target input count = 1 + K
executed_after               = P + 1 + K
materialized_after           = P + 1 + A
output_tokens                = A accepted drafts + 1 recovery/bonus token
```

The recovery/bonus token has not been evaluated. All rejected suffix slots stay
invisible. If the engine keeps only m output tokens because of EOS/stop/length
limits, at most `P + 1 + min(A, m)` tokens of this forward's state can remain
valid; m may be zero on cancellation. Per-group checkpoints may impose a lower
retained length. `Settle`/cache commitment records the exact decision before
reusing state or publishing a prefix-cache entry.

Example: P=10, K=3, A=1 evaluates 14 positions but validates a prefix of length
12 and produces `[d0, recovery]`. A following decode consumes `recovery` at
position 12. Keeping only `d0` at a stop boundary retains at most length 12.

Draft-state advancement is reconciled separately: truncate accepted prefix,
restore a checkpoint, or recompute from committed tokens. Target acceptance does
not certify a draft model's recurrent state. TP/PP+speculation requires agreed
verification-row transport, proposal placement, and per-stage rollback coverage.
Tree-shaped proposals, diffusion decoding, and cross-tokenizer speculation remain
explicit exclusions from this chain-verification profile.

## 15. Multimodal, encoder-decoder, pooling, and artifact APIs

### Task envelope and input bindings

```cpp
using ExecutionTask = std::variant<ExecutionBatch, EncoderBatch,
                                   PoolingBatch, MaintenanceStep>;
struct ExecutionRequest {
  ExecutionProfileId profile;
  std::optional<DomainEpoch> domain;
  ExecutionTask task;
};

struct DecoderExtensions {
  std::vector<DependentInput> dependencies;
  std::optional<EmbeddingBindings> embeddings;
  std::optional<PositionBindings> positions;
  std::vector<EncoderBinding> encoder_outputs;
  std::vector<GroupStateBinding> state;
  std::vector<ArtifactExport> exports;
};

struct EncoderItem {
  EncoderItemId id;
  ModelIdentity encoder;
  Modality modality;
  std::vector<OwnedTensorRef> inputs;
  PreprocessingFingerprint preprocessing;
  AdapterRef adapter;
};

struct EncoderBatch {
  uint64_t engine_step;
  std::vector<EncoderItem> items;
};

struct EncoderBinding {
  uint32_t sequence_index;
  ArtifactRef encoded;
  EncoderUse use; // Embedding replacement or cross-attention source.
  std::vector<PlaceholderRange> ranges;
};
```

Encoding consumes preprocessed tensors; file fetching, image decoding, tokenizing,
and user-facing media validation belong upstream. `EncoderBatch` can run with
zero decoder tokens. Its result contains encoder artifacts with exact shape,
dtype, mask/length metadata, producer model/adapter identity, and content plus
preprocessing fingerprints. Cache identity includes all of these, not just a
media hash. The encoder may be a module of the target or an attached auxiliary.

`EmbeddingBindings` supplies owned `[N,H]` embeddings and a unique mapping to
flat input rows. Unmapped rows use token embeddings. Replaced rows need no valid
vocabulary ID; their placeholders are explicitly tagged and cannot accidentally
reach the token embedding kernel. `PositionBindings` supplies a model-declared
schema (scalar, multimodal axes, etc.) with tensor dtype/dimensions and an exact
row mapping. Positional coordinates may differ from logical cache indices;
the model schema defines the relationship and validation.

For image/video/audio replacement, each `PlaceholderRange` identifies absolute
sequence positions, the corresponding artifact rows, and modality layout.
Chunks may bind just an intersecting range while pinning the parent artifact.
Reject gaps, overlaps, incorrect feature counts, stale revisions, or unsupported
splits of a model-required atomic modality region before launch.

Encoder-decoder bindings identify source sequence length/mask and immutable
cross-attention state separately from decoder self-KV. Decoder position zero is
not the encoder length. The initial decoder BOS/start token is explicit. Prefix
reuse must match encoder identity/content as well as decoder tokens and adapter.
Streaming audio models additionally need a declared chunk-state schema; arbitrary
media streaming is not implied by ordinary `EncoderBatch` support.

### Pooling and non-generation tasks

```cpp
struct PoolingBatch {
  uint64_t engine_step;
  ModelInputs inputs; // Packed IDs/embeddings plus attention-mask semantics.
  std::vector<PoolingSpec> outputs;
  std::optional<PoolingStateLease> continuation;
};

struct PoolingSpec {
  uint32_t sequence_index;
  PoolingTask task;   // Embedding, classification, reward, or token output.
  PoolingMethod method; // Model-defined, last, first/CLS, mean, or all.
  std::optional<uint32_t> dimensions;
  bool normalize;
  bool final_chunk;
};
```

`ModelInputs` carries sequence keys/history revisions and states whether attention
is causal or bidirectional, sequence boundaries, masks, modality bindings, and
any needed state leases. A pooling
model does not have to accept causal decoder cache tables. Output is an artifact
with task, sequence key, shape, dtype, and normalization/dimension metadata;
classification/reward heads are invoked only if loaded and advertised.

Chunked causal mean pooling carries a versioned `(sum, count)` accumulator;
last pooling carries its last selected hidden row; all-token output retains or
streams bounded token chunks in order. Exactly one finalization produces the
final request result. Bidirectional encoders generally require the complete
sequence; they must reject independent chunk pooling unless the model advertises
an exact equivalent algorithm. Requested dimension reduction/normalization must
match a supported model transform, not silent truncation of arbitrary vectors.

### Artifact lifetime and output size

`ArtifactRef` is an opaque, reference-counted identity containing producer
incarnation/runtime epoch, artifact kind, schema, shape/dtype, and readiness
token. It pins the underlying bytes and never exposes a process-local pointer
as a transport identity. Encoder features, raw logits, selected hidden layers,
pooling outputs, and draft probabilities use this mechanism.

The artifact API is `Describe(ref)`, `Read(ref, row_range, deadline)` for bounded
host materialization, and `Release(ref)`; compatible downstream operations can
consume it directly on device. A move-owned wrapper releases on destruction.
An export request fixes rows/layers, normalization stage, dtype, and byte limit
before execution. Raw vocabulary logits require an explicit capability and
budget. Reads are chunked if necessary; they cannot secretly materialize an
unbounded `[all_tokens, vocabulary]` tensor on the scheduler.

Each `ArtifactResult` contains its requested operation index, owner identity
(`SequenceKey` plus history revision, or `EncoderItemId`), and `ArtifactRef`.
Encoder/pooling results leave generation samples empty; their identity is never
inferred from a dummy sequence-progress or sampled-token entry.

These operations are exposed through `Executor::artifacts()`. `Describe` returns
metadata without synchronizing a device; `Read` returns an owned host tensor and
observes the same nonterminal caller-deadline rule as ticket waits. `Release`
is idempotent for the same local handle and drops only that ownership reference.
Rank-local tensor references and engine-visible artifact references have distinct
types so a local allocation cannot accidentally be used as a cross-process ID.

External tensors must provide an ownership lease and readiness fence. An
unowned span is copied into owned staging before acceptance returns. Adapters,
encoder cache entries, and source buffers remain pinned through every consumer.
Sleep/reset refuses pinned affected artifacts or uses explicit invalidation after
all readers are safe; the engine receives which handles became invalid.

## 16. Cache groups, recurrent state, and commit atomicity

```cpp
using StateLayout = std::variant<FullAttentionLayout, WindowAttentionLayout,
                                LatentAttentionLayout, RecurrentLayout,
                                CrossAttentionLayout>;

struct CacheGroupDescriptor {
  CacheGroupId id;
  StateLayout layout;
  std::vector<LayerId> layers;
  std::vector<WorkerId> storage_workers;
  NumericalFormat format; // Dtype, packing, quantization/scales, alignment.
};

struct GroupStateBinding {
  uint32_t sequence_index;
  CacheGroupId group;
  std::variant<PagedRangeBinding, RecurrentBinding,
               CrossAttentionBinding> binding;
};

struct StateCheckpoint {
  SequenceKey sequence;
  CacheGroupId group;
  Position valid_through;
  SnapshotRef snapshot;
};
```

Plain full attention continues to use `cache_tables`. A group must be bound
exactly once, through that representation or `extensions.state`. A
`PagedRangeBinding` carries logical start positions, page references, and valid
read/write intervals; it does not assume every past token remains resident.
Latent attention declares its packed latent/positional shapes, and cross-attention
binds the encoder-derived immutable state. Rank storage workers reflect PP/TP
placement and may use different physical byte sizes.

`StateLease` is the generalized reservation authority for page ranges, recurrent
snapshots, and cross-attention state; `CacheLease` is its causal-batch wrapper.
Both retain the same checked pool epochs, usage rights, readiness fences, and
reference-counted lifetime. `SnapshotRef` and `PoolingStateLease` additionally
carry a schema and logical version; they cannot alias writable committed state.
Physical allocation APIs remain cache-manager concerns, not executor methods.

For a causal window of W including the current token, the first query at P needs
old positions `[max(0,P-W+1), P)`. Subsequent queries consume relevant preceding
tokens from the new chunk. Retain the union needed by every query and any pending
reader/checkpoint. Circular-buffer reuse must be ordered behind those readers;
speculative or asynchronous lookahead adds retention capacity. Full-attention
groups in the same model still retain their full required prefix.

`RecurrentBinding` supplies an immutable input snapshot at exactly the claimed
prefix and exclusive output state storage. Mutation of the only committed
snapshot is forbidden. Chunk execution returns checkpoints at declared valid
positions. A capability declares whether arbitrary token checkpoints, aligned
checkpoints, replay, or no rollback is supported. Speculation with a recurrent
model is rejected unless every possible accepted boundary can be restored or
recomputed exactly. Merely shortening a token count does not restore recurrence.

The cache manager commits or aborts all groups for a sequence atomically at the
visibility layer. It publishes a new sequence-state version only when every
group supplies compatible validity at that boundary. Failure in one group
cannot leave another group's longer suffix visible. Physical writes need not
be transactional because their leases keep provisional storage isolated.

Prefix identity includes model weights/config, adapter revision, canonical token
or embedding content, position policy, encoder content, cache numerical format,
and state-layout version. Reusing a numeric page ID is never proof of identity.
Preemption resumes from a retained complete state checkpoint or recomputes; it
does not mix a recurrent snapshot from one prefix with attention pages from
another. Quantized cache transfers include required scales and layout metadata.

## 17. Cache transfer, offload, and maintenance-only steps

```cpp
struct TransferPlan {
  TransferId id;
  TransferDirection direction; // Import or export.
  ConnectorRef connector;
  RemoteStateManifest remote;
  StateLease local_state;
  std::vector<StateRange> ranges;
  TransferPolicy policy; // Whole-state or explicitly supported layerwise.
};

struct MaintenanceStep {
  uint64_t engine_step;
  std::vector<TransferPlan> transfers;
  std::vector<TransferId> observe;
  std::vector<ArtifactRef> release;
  bool participate_in_domain;
};

struct TransferProgress {
  TransferId id;
  TransferPhase phase; // Pending, ready, failed, cancelled.
  std::vector<StateRange> completed;
  std::optional<TransferReceipt> receipt;
  Status status;
};
```

`RemoteStateManifest` contains peer identity, model/adapter/content fingerprints,
state layout/dtype/scales, layer placement, logical token ranges, lengths,
connector protocol version, and destination mapping requirements. Handshake is
an explicit connector control operation. Different TP/PP layouts require a
negotiated reshard/convert implementation; matching model names are insufficient.
Host/storage offload uses the same contract with a different connector target.

Submission accepts transfer work and retains its leases. A maintenance result
may report `Pending`; it does not claim transfer completion. Connector service
makes progress independently of decoder batches, and later maintenance polls
report terminal state. Terminal records are retained until explicit acknowledgment
or connector shutdown. Duplicate transfer IDs with identical manifests are
idempotent while retained; conflicting reuse is rejected. After reconnect,
query the receipt before deciding to retry an ambiguous transfer.

Exports normally require committed state. A layerwise implementation may stage
provisional bytes, but cannot publish the manifest until an engine commit receipt
certifies the exported prefix. Imports write exclusive reserved storage and
become visible only after every required layer/rank completes and validates.
Layerwise reads can overlap import only through per-layer readiness fences and
explicit failure propagation. Cancellation must await or abort DMA safely before
recycling source/destination pages. A partial import is discarded, never counted
as a prefix hit.

Disaggregated prefill/decode sequence:

1. Producer prefills and commits prefix length P. It may also sample first token x.
2. Export a manifest for the committed state through P, keeping source pages pinned.
3. Consumer imports into its own reservation and verifies compatibility/receipt.
4. Engine transfers canonical history, sampling/RNG index, and x if already sampled.
5. Consumer feeds x at position P. If no x/logits were transferred, explicitly
   recompute the last prompt row to obtain the first next-token distribution.

The KV payload alone does not encode whether a first output was sampled. Import
readiness is a transfer property; request completion and client streaming remain
engine properties. Transfer-only steps must not invoke the decoder accidentally.

## 18. Adapters, runtime controls, and quiescent maintenance

```cpp
using ControlCommand = std::variant<InstallAdapter, RemoveAdapter, PinAdapter,
    UnpinAdapter, ListAdapters, ConnectorHandshake, AcknowledgeTransfers,
    CancelTransfer, Quiesce, Resume, ResetCache, Sleep, Wake,
    StartTrace, StopTrace, ExportWeights>;
```

`Control` is a typed asynchronous operation returning `ControlTicket`, whose
poll/wait semantics match execution tickets and whose result is a tagged
`ControlResult` with command ID, affected epochs/handles, and rank acknowledgments.
The worker protocol may implement this with RPC, but the public API accepts no
arbitrary method strings or serialized callbacks.

| Command | Preconditions and result contract |
| --- | --- |
| Install adapter | Immutable artifact identity, base-model compatibility, rank/dtype/target-module manifest, and budget. Prepare every affected rank before publishing a revisioned handle; failed partial installs are cleaned up. |
| Pin/unpin/remove/list adapter | Pin is residency policy; execution separately holds a usage reference. Remove waits for users or returns busy and never evicts an in-flight adapter. List reports coherent installed revisions, not a rank-local guess. |
| Quiesce | Close admission and finish accepted execution, unresolved forwards, dependencies, and transfers. Caller must resolve/discard/settle outstanding logical work; timeout reports the blockers. Returns a quiescence token. |
| Resume | Requires a valid quiescence token and fully ready runtime. Opens admission after all ranks acknowledge current epochs. |
| Reset cache | Requires quiescence and released affected state/artifact leases. Selects KV, encoder, or all state. Bumps corresponding epochs and returns invalidation metadata. |
| Sleep | Requires quiescence. Selects weight/cache/workspace residency policy, preservation/offload requirements, and memory target. Returns exactly what was retained, offloaded, or invalidated. |
| Wake | Restores matching resident weight revision from retained/offloaded storage, rebuilds invalidated pools/graphs, and returns `ReadyInfo`. Ends quiesced; explicit resume follows engine reconciliation. |
| Start/stop trace | Names bounded rank/session scope, duration/iterations, and metrics. Uses ordered control boundaries; tracing does not issue unscheduled model collectives. |
| Export weights | Explicit destination and immutable weight revision, with quiescence or supported snapshot isolation. Completion includes rank/shard manifest; not a sampling-path side effect. |

Adapters are bound per sequence and propagated to encoder/draft modules only if
those modules declare compatible support. Graph cache keys include adapter
layout/rank/residency dependencies. A warmed graph may use dynamic adapter IDs
only if its implementation safely supports them. Cache identity always includes
adapter revision, so switching an adapter cannot reuse another adapter's KV.

Quiescence differs from ordinary `Drain`: it remains closed until resume and
accounts for retained logical state, not just launched kernels. Resolution,
settlement, discard, acknowledgments, and release remain callable while admission
is closed so shutdown/maintenance cannot deadlock on their own prerequisites.

Sleep may invalidate device addresses. Old graph executables and IPC descriptors
are revoked before reuse, and all externally visible pool/artifact epochs are
updated. If any rank fails a destructive control transition, the executor enters
`Failed`; it cannot advertise a mixture of old and new state as ready.

Base-weight replacement, vocabulary changes, and topology resizing use a new
`LoadedWorkerGroup` and executor incarnation. Live online weight synchronization
and elastic worker membership are explicitly deferred. The API still supports
sleep/wake of the same revision, adapter lifecycle, tracing, and weight export
without pretending these operations are ordinary model forward calls.

## 19. Protocol robustness, compatibility, and operational limits

### Ownership and wire representation

C++ snippets describe a logical interface, not a stable binary ABI. Out-of-process
workers negotiate a versioned schema before attachment. Messages carry executor
incarnation, runtime/topology epochs, command/submission ID, and operation kind.
Lengths and enums are validated before allocation; unknown mandatory fields or
unsupported operation versions fail negotiation. IDs, leases, and tensor handles
are serialized identities, never raw pointers or `std::variant` memory layouts.

Use bounded control messages for metadata and a separate shared-memory/device/
network data path for large tensors. A byte payload has an explicit owner, size,
checksum where transport requires it, and ready/release acknowledgment. Transport
disconnect or ambiguous send after acceptance becomes a ticket failure; do not
silently dispatch the same model work again. Late responses from old incarnations
are ignored for computation but still trigger safe transport cleanup.

### Resource accounting and deadlock prevention

Admission charges worst-case retained bytes, including selected logits, draft
probabilities, PP activations, encoder artifacts, state snapshots, graph padding,
transfer pins, and host result queues. Allocating only from input-token counts
underestimates these workloads. Credits distinguish accepted work, active device
slots, completed unconsumed results, and externally retained artifacts.

Pre-reserve completion/control credits separately from new-work credits.
Deferred resolution must have enough workspace already reserved; pipeline sends
must not hold all buffers while awaiting downstream allocations; transfers must
not pin every page needed to complete an admitted batch. If that cannot be
guaranteed, reject new admission with a resource-specific `ResourceExhausted`.

Persistent worker-side metadata may cache immutable request descriptors keyed
by `(sequence, history_revision, model/adapter revision)`. The public contract
remains a complete logical step; transport compression can send a known version
plus deltas only with an acknowledged base version and a full-snapshot fallback.
A missing/stale base is rejected before launch and resynchronized; retirement
is acknowledged after readers finish. This optimization cannot become a second
authority for committed history.

Distributed admission uses a prepare/launch protocol: the coordinator assigns a
submission identity, prepares worker-local allocations and immutable metadata,
collects readiness, then issues the common launch order. Failure before launch
releases prepared resources; uncertainty after a launch message is treated as
accepted work and fails through a ticket/group abort. This is not transactional
GPU execution: a disconnected rank cannot be assumed to have done nothing.

### Completion and errors across operation kinds

| Event | Meaning | What it permits |
| --- | --- | --- |
| Accepted | All logical admission resources reserved; ticket returned. | Caller may release moved input objects. |
| Device-ready | Producer's named device outputs are consumable through fences. | A validated dependent operation may read them provisionally. |
| Host-ready | Compact result is readable; ordinary batch device work is complete. | Engine can validate and decide commit. |
| Settled | Engine/cache receipt fixes the accepted output/state prefix. | Matching descendants and prefix publication can be committed. |
| Resources-safe | All kernels, readers, DMA, and transport accesses have ended. | Lease authority may recycle storage once logical references are gone. |

These events are related but not interchangeable. A fatal status may precede
resources-safe; a pending transfer result may be host-readable long before its
DMA completes; an artifact may remain pinned after its producing batch ends.
Ordinary host-driven generation needs no explicit `Settle` unless it exports
provisional dependencies; cache commitment is sufficient for its next full batch.

Errors include operation ID, phase (`Validate`, `Prepare`, `Forward`, `Resolve`,
`Transfer`, `Control`), rank/stage if applicable, status code, recovery class
(`Rejected`, `BatchFailed`, `GroupFailed`), and resource-safety state. Preserve
this structured payload through remote transport. Never parse free-form error
text to decide whether cache pages can be freed.

Recoverable batch failures return no partial sequence success. A scheduler can
retry unaffected sequences only from known committed state and a new reservation.
Worker/device/collective loss poisons the group and pending dependent work.
Numerical diagnostics distinguish invalid logits from transport failure, and
the configured policy determines whether a numeric failure poisons just the
batch or the model replica. Automatic deterministic retry is not assumed.

### Explicit exclusions and compatibility gates

The design covers the workflows in section 11, not every upstream experimental
method. Tree speculation, cross-tokenizer draft translation, diffusion execution,
arbitrary Python logits callbacks, training/backward/optimizer steps, live base-
weight replacement, and elastic topology changes are outside the defined profile.
Audio/video chunk streaming requires a model-specific state schema. Feature
combinations involving recurrent rollback, PP+speculation, LoRA+graphs,
multimodal+prefix reuse, EP idle work, and quantized transfer must each have an
explicit tested admission profile; otherwise reject them with the missing contract.

The following combinations are explicit design targets; enabling their component
capabilities is insufficient without satisfying the joint conditions:

| Combination | Required joint contract |
| --- | --- |
| TP×PP + continuous batching | Stage schemas, microbatch row remapping, one output owner, common collective schedule, bounded activation credits |
| PP + speculative verification | Last-stage verification rows, target/draft placement, per-stage suffix invalidation, returned accepted lengths |
| Speculation + grammar/penalties | Per-hypothetical-prefix processor state and masks; actual q contract for stochastic verification |
| Async decode + stop/cancellation | Provisional dependency DAG, ordered settlement, descendant invalidation, reserved rollback capacity |
| Hybrid recurrent state + speculation | Snapshot/replay coverage for every possible commit boundary across all state groups |
| LoRA + graph replay | Pinned adapter revision, supported dynamic/static adapter layout, graph-key invalidation rules |
| Multimodal + prefix reuse / PP | Encoder-content and position identity, chunk placeholder mapping, cross-stage artifact transport/lifetime |
| Transfer + quantization / different TP | Scales/layout manifest, explicit conversion/resharding, all-rank import receipt before publication |
| EP + empty local work / graphs | Domain admission, explicit idle participation, matched collective/padding shapes without live-cache writes |

An implementation must name which combination failed admission. It cannot claim
support by silently disabling grammar, changing a sampler, dropping adapters,
truncating context, or turning off requested score outputs. `Prefer` graph policy
is an explicitly authorized eager fallback; other semantic changes require the
engine to submit a different valid plan.

Choose and record a vLLM commit when implementing numerical parity tests.
Sampling transforms, RNG algorithms, score spaces, quantization kernels, and
graph padding can differ while an API appears similar. Report exact-ID parity
only where guaranteed; use numerical tolerance or distributional tests elsewhere.

## 20. Implementation milestones and acceptance tests

Implement the same proposed contract in stages; never advertise a capability
just because its message type exists:

| Milestone | Required operations | Validation target |
| --- | --- | --- |
| M1 Core | Loaded handoff, profile/init, eager causal generation, score rows, leases, greedy and supported stochastic policies, tickets/drain | U01–U09 on one device; basic failures and ownership |
| M2 Distributed/performance | TP, PP, multi-node transport, graph dispatch, bounded overlap, domain idle protocol | U10–U14, U24; fault injection and collective ordering |
| M3 Broader model tasks | Encoder/multimodal bindings, pooling, adapters, hybrid-state groups | U16–U20; model-specific input/state contracts |
| M4 Advanced workflows | Deferred resolution, device dependencies, chain speculation, transfer/offload, controls | U07, U14–U15, U21–U26; settlement and quiescence |

Deferred output resolution may ship earlier than M4; coupled EP/context execution
requires its own backend tests before M2 advertises those modes. These are
implementation priorities, not changes to the public API's intended coverage.

Contract tests should include:

1. **Admission:** invalid tokens, integer overflow, duplicate keys/rows, wrong
   model/adapter/epoch, missing groups, contradictory capabilities, and OOM
   reservation launch no model work or live-cache writes.
2. **Causal correctness:** chunked versus whole prefill; cached-prefix extension;
   mixed-batch reordering; last-row selection; score-only and cache-only paths;
   full-prefix hits with missing logits or prompt scores.
3. **Progress:** sampled token is not computed KV; EOS and maximum-output limits
   preserve correct lengths; branch keys isolate RNG/state; stale generations
   cannot commit after cancellation/preemption/recompute.
4. **Sampling:** prompt versus output penalty history, minimum length, banned
   sequences, all-masked/NaN logits, tied argmax, selected/top/ranked score spaces,
   batch-order invariance of RNG, and stochastic distribution checks.
5. **Lifetime:** delayed H2D/D2H, dropped/expired tickets, unresolved forwards,
   caller wait timeouts, retained artifacts, and DMA completion prevent early
   buffer/page reuse. Completed results remain valid after scratch reuse.
6. **TP/PP:** one coherent output owner, last-stage sampling, uneven stage costs,
   reordered transport arrival, full activation-credit queues, per-stage cache
   mapping, and worker loss during a collective/send/receive.
7. **EP/context:** idle participants, globally agreed epochs, zero real tokens,
   sharded context causality, and coordinator failure without unmatched collectives.
8. **Deferred/async:** grammar arrives late or stale, invalid resolution is retried,
   double resolve fails, dependency cycle is rejected, producer fails, parent
   stop invalidates launched descendants, and control credits prevent deadlock.
9. **Speculation:** K=0, variable K, first rejection, all accepted plus bonus,
   deterministic q, stochastic rejection/recovery, per-prefix penalties/masks,
   stop inside accepted prefix, draft rollback, and rejected KV invisibility.
10. **Multimodal/encoder:** repeated encoder cache hits, chunk-boundary placeholders,
    invalid dimensions/axes, embedding-only rows, cancelled consumers, source/
    decoder length separation, and changed preprocessing/adapter identity.
11. **Pooling:** every supported task and mask convention, chunk-state finalization,
    bounded all-token output, normalization/dimensions, and rejection of invalid
    bidirectional chunk decomposition.
12. **State/cache:** shared-tail COW, window wrap with in-flight readers, mixed
    group layouts, missing recurrent checkpoints, atomic visibility across groups,
    quantized scales, stale snapshots, and group-specific PP storage.
13. **Transfer:** no-forward progress, layerwise readiness, producer first-token
    handoff, incompatible topology/format, partial import, reconnect/duplicate ID,
    cancellation during DMA, source retention, and receipt acknowledgment.
14. **Controls:** adapter prepare failure on one rank, remove while pinned,
    quiescence blocked by retained work, sleep/wake epoch invalidation, graph
    recapture, trace start/stop ordering, and shutdown timeout followed by retry.
15. **Graphs:** eager/replay parity for supported profiles, padding isolation,
    overlapping graph slots, adapter changes, unavailable shapes under `Require`,
    and consistent routes within each communicating group.

Use a deterministic fake runtime with controllable fences and injected worker/
transport failures for ownership tests; use real backends for numerical,
collective, transfer, and graph tests. Compare TP/PP outputs with single-device
references within documented tolerances. Measure TTFT, inter-token latency,
throughput, host synchronization count, memory peaks, and queue bounds without
changing semantics to obtain a benchmark result.

This file specifies the design and acceptance criteria. No runtime features or
acceptance tests are claimed to be implemented or executed by this document edit.
