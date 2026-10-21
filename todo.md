# InferX vs vLLM: Gap Analysis and Engineering TODO

> Analysis date: 2026-10-21. This document is the only change made for this
> audit; no source code was modified. No root `todo.md` existed before this
> file, so there were no unrelated TODO entries to preserve.

## 1. Project Overview

InferX is a native C++23/CUDA LLM inference engine and OpenAI-compatible
serving stack. Its current architecture is:

- **Scheduling:** `Scheduler` performs FCFS admission, continuous batching,
  chunked prefill, token-budgeted steps, prefix-cache-aware admission, and
  recompute preemption (`include/inferx/engine/scheduler.h`,
  `src/engine/scheduler.cc`).
- **Execution:** `ModelRunner`, `Worker`, and `InputBatch` own request state,
  build a flat step batch, upload metadata, run the model, sample, and return
  sampled ids to the scheduler (`src/engine/model_runner.cc`,
  `src/engine/input_batch.cc`, `src/engine/worker.cc`).
- **KV cache:** a single byte-denominated paged pool (`KvBlockPool`), a
  request-level cache facade (`CacheManager`), per-retention-policy group
  managers, a recurrent-state pool, SHA-256 chained prefix hashing, and LRU
  eviction (`include/inferx/cache/`, `src/cache/`).
- **Attention:** a unified spec/backend runtime with FlashInfer dense paged
  attention and a CPU reference oracle (`include/inferx/ops/attention/`,
  `src/ops/attention/`).
- **Models:** Qwen3 dense, Qwen3-MoE, and Llama causal language models, BF16
  execution, safetensors checkpoints
  (`src/models/registry.cc`, `src/models/qwen3/`, `src/models/qwen3_moe/`,
  `src/models/llama/`).
- **Serving:** Boost.Asio/Beast HTTP/1.1, `/v1/completions`,
  `/v1/chat/completions` (Qwen3 text template), SSE, API keys, prompt
  preprocessing workers, and a small Prometheus text endpoint
  (`src/server/`, `include/inferx/server/`).
- **Parallelism:** in-process NCCL tensor parallelism; topology, sharding,
  worker-IPC, and pipeline/data/expert-parallel types exist but are not fully
  wired into the engine (`src/dist/`, `src/engine/model_runner.cc`).

Implemented and working in the current tree include continuous batching,
chunked prefill, recompute preemption, paged KV allocation, full-block prefix
caching, FlashInfer attention, greedy and sampled decoding, Qwen3 chat, SSE
streaming, cancellation, and single-process tensor parallelism.

Intentional scope boundaries (not automatically defects): CUDA-only model
execution, BF16 model execution, local safetensors loading, a deliberately
small architecture registry, and a text-generation-first server. The gaps
below are judged against vLLM where they matter to this project's actual
serving scope.

### Performance status: Needs Investigation

There is no benchmark of the current HEAD in this audit. Historical artifacts
exist, but they are pre-refactor and must not be presented as current results:

- `benchmarks/qwen3/REPORT.md` records an older 24-case offline matrix in
  which "None of the 24 fully measured configurations exceeds vLLM
  throughput"; it also records failing exact greedy agreement.
- `benchmarks/qwen3/OPTIMIZATION_20260922.md` records a later experimental
  candidate at 1.053x aggregate throughput versus a fresh vLLM 0.29.0 run,
  with 4/24 cases slower and exact token equivalence still failing.
- `benchmarks/qwen3/serve_results/rerun_20260923/REPORT.md` records matched
  eager HTTP runs against vLLM 0.29.0 where InferX had higher output
  throughput but higher mean TTFT in 0/9 and 0/18 comparable cases (i.e. vLLM
  was faster to first token in every case).
- `docs/bugs.md` records open correctness, concurrency, and KV-pressure issues
  from 2026-09-22/23; some may have moved since, so the source is authoritative.

These are useful baselines, not a current performance claim. Re-run the full
offline and HTTP matrices after any of the performance work below.

## 2. Comparison Scope

- **Primary reference:** vLLM **0.29.0**, installed in
  `python/.venv/lib/python3.12/site-packages/vllm`. Its configuration surface
  was inspected for cache/scheduler/parallel/speculative/LoRA support. The
  audit did not execute vLLM or reproduce its benchmarks.
- **Design-reference snapshots in this repository:** `docs/attention_design.md`
  and `docs/inference_server_design.md` reference vLLM main commit
  `428cd0cb4a42750f5339d08bb67d534cf8051429` and v0.11.0 commit
  `b8b302cde434df8c9289a2b465406b47ebab1c2d`. Comments and artifacts may
  therefore describe features newer than 0.29.0.
- **Method:** source inspection, header/config inspection, existing test and
  benchmark artifacts. No claim that "vLLM does X" is made from memory alone
  where the installed 0.29.0 source or the repository's pinned references
  could be checked. Items not verifiable from the tree are marked
  **Needs Investigation**.
- **Comparison assumptions:** same GPU, BF16 weights, greedy sampling, local
  token-id prompts, and comparable KV/scheduler budgets. The project is not
  expected to clone every vLLM backend; gaps that are outside its stated
  scope are marked as such.

## 3. Limitations Summary

Priority combines user-visible impact, risk to correctness, dependency order,
and fit with the current Qwen3/BF16/CUDA scope.

| ID | Area | vLLM 0.29.0 | InferX today | Priority |
|---|---|---|---|---|
| P0-1 | Scheduler liveness | Progress-preserving preemption/reservation | Can repeatedly emit empty steps when all live requests need a block | High |
| P0-2 | Context limits | Validates prompt + output against `max_model_len`; returns context error | `--max-model-len` is unused; overlong requests fail inside the runner and can kill the whole step | High |
| P0-3 | HTTP concurrency | Production async server with per-connection serialization | Acceptor/session operations run on a raw multi-threaded `io_context` without strands | High |
| P0-4 | Shutdown/health | Graceful shutdown and engine-aware health | Grace timer can be defeated by engine drain/join; `/health` does not reflect engine death or stalls | High |
| P0-5 | Admission vs. stop | Admission and engine shutdown are synchronized | `Submit` can succeed after the engine has exited | High |
| H1 | CUDA graphs | Graph/compile path is a core production optimization | `CaptureGraphs` returns Unimplemented; TP rejects graphs | High |
| H2 | Metadata hot path | Device/persistent metadata, asynchronous staging | Every metadata array is H2D-uploaded and stream-synchronized per group per step | High |
| H3 | Per-step synchronization | Overlapped scheduling/execution | Full stream synchronize and pageable D2H sampler copy every step | High |
| H4 | Parallelism | TP, PP, DP, EP, context-parallel execution | Only in-process TP is wired; unsupported axes are silently ignored | High |
| H5 | Sampling/API controls | logprobs, n > 1, structured outputs, logit bias, beam, tools | Many accepted in `SamplingParams` internally but rejected by the HTTP layer; no grammars/logprobs | High |
| H6 | Speculative decoding | n-gram, draft model, EAGLE/Medusa/MTP paths | Configuration header and design doc only; no engine path | Medium |
| H7 | LoRA/multi-adapter | Multi-LoRA serving, adapter caching, batched adapters | Configuration header only; no loader, kernels, API, or batching | Medium |
| H8 | Quantization | FP8, AWQ/GPTQ and other checkpoint/KV quantization | Quantized checkpoints rejected; only BF16 weights/activations; FP8 KV config inert | Medium |
| H9 | Model/attention coverage | Broad architecture and attention coverage | Qwen3 dense/MoE and Llama only; sliding-window rejected; MLA/GDN/sparse are infrastructure without models | Medium |
| H10 | MoE performance/EP | Grouped expert execution and expert parallelism | Per-layer host sync, per-expert loop, all experts resident per rank | Medium |
| H11 | Prefix cache tiers | Prefix cache plus CPU/disk/remote KV offload | In-GPU prefix cache only; no offload, swap, or transfer tiers | Medium |
| H12 | Observability | Queue/KV/cache/TTFT/token/spec/LoRA metrics | Only HTTP and prompt-pool metrics; scheduler/cache stats are not exported | Medium |
| M1 | OpenAI surface | Completions/chat/embeddings/tokenize/detokenize and rich fields | Completions/chat only; only Qwen3 template; many fields rejected | Medium |
| M2 | Multimodal | Image/audio/video preprocessing and model execution | Types exist but API and prompt processor reject multimodal input | Medium |
| M3 | Tokenizer/loader | HF tokenizers, SentencePiece, GGUF/PT/sharded formats | `tokenizer.json` only; safetensors only; SentencePiece rejected | Medium |
| M4 | Chunked-prefill config | Configurable behavior | Config flag is accepted but ignored; scheduler always chunks | Medium |
| M5 | Scheduling policy | FCFS and priority policies, configurable queue | FCFS only; priority/fair policy fields are unused | Medium |
| M6 | Preemption | Recompute and swap/offload | Recompute only; no host/swap path | Medium |
| M7 | Perf of prefill sampling | Avoids unnecessary logits/sampling on incomplete prefills | Computes LM-head logits and samples every scheduled prefill chunk, then discards | Medium |
| M8 | Sampling metadata cost | Batched device-side metadata | Copies full token history and uploads deduped history every step even without penalties | Medium |
| M9 | TP collectives | Optimized collectives/fusion | One NCCL all-gather per logits row; no communication overlap | Medium |
| M10 | Attention selection | Capability-checked backend selection | `Env` omits SM/graph requirements; FlashInfer head dims are not declared; unsupported geometry fails at launch | Medium |
| L1 | Worker IPC / CLI | Distributed execution entry points | Worker pool/IPC exists but no `inferx worker` command or controller integration | Low |
| L2 | Benchmark CLI | Latency/throughput/online benchmarks | C++ `bench latency` and `bench throughput` are unimplemented; external Python benchmarks exist | Low |
| L3 | Client/tooling | Rich CLI/OpenAI clients | `chat`, `complete`, `run-batch`, `collect-env`, `launch render` are stubs | Low |
| L4 | CPU backend | CPU serving path | CLI accepts `--device cpu`, but model execution requires CUDA | Low |
| L5 | Cache/config hygiene | Validated, honored config | `cache_dtype`, `sliding_window`, `calculate_kv_scales`, `prefix_caching_hash_algo`, `max_model_len`, and `seed` are declared but not all honored | Low/Medium |

## 4. Detailed Limitations

### 4.1 Correctness, Liveness, and Server Reliability

#### P0-1 — Scheduler livelock when every live request needs KV growth (High)

- **vLLM:** The scheduler reserves capacity for the scheduled work, preempts
  or swaps victims when the cache cannot grow, and does not silently produce
  endless empty steps while requests remain.
- **InferX:** The running-request loop reserves only the next chunk/token. If
  `Reserve` fails and no newer victim can be preempted, it executes `continue`
  (`src/engine/scheduler.cc:59-77`). Waiting admission similarly breaks on
  failure (`src/engine/scheduler.cc:126-137`). A plan can therefore be
  successful but empty while `HasRequests()` is true.
- **Evidence:** `benchmarks/qwen3/serve_results/rerun_20260923/CONCURRENCY.md`
  contains a deterministic scheduler probe that reports 99 empty steps and a
  live HTTP probe where two concurrent requests both time out while the
  serial control succeeds; `/health` still returned 200 afterward.
- **Why it matters:** This is a serving livelock, not just a slow path. No
  terminal event is produced and the engine thread keeps stepping an empty
  plan.
- **Approach:** Reject requests that cannot fit individually, reserve
  headroom for at least the next decode token, preempt victims before the
  current request is attempted where necessary, and treat repeated
  no-progress plans as an explicit failure or health transition. Add a
  "nonempty scheduler produced N empty steps" invariant and metric, plus a
  scheduler/HTTP regression test. Ensure the engine event loop wakes/terminates
  rather than busy-spinning.

#### P0-2 — Context length and `--max-model-len` are not enforced at admission (High)

- **vLLM:** Validates prompt length plus requested output against the model
  context limit and returns a structured context-length error before GPU
  work.
- **InferX:** `ModelConfig::max_model_len` is declared and exposed via
  `--max-model-len` (`include/inferx/config/model_config.h:32-35`,
  `apps/cli/args/model_config_args.h:66-70`) but is not passed into
  `ModelRunner`/`AttentionConfig`. The runner sizes against
  `spec.max_position_embeddings` instead
  (`src/engine/model_runner.cc:240-241`). Prompt length is checked only when
  `InputBatch::Add` runs inside the step
  (`src/engine/input_batch.cc:29-32`), and appended output hits the limit in
  `InputBatch::AppendToken` (`src/engine/input_batch.cc:114-115`).
- **Failure mode:** `ModelRunner::RunStep` propagates the `AppendToken` error
  from `ApplyAcceptedTokens` (`src/engine/model_runner.cc:429-435`);
  `EngineGateway` treats a runner failure as fatal and calls `FailAll`
  (`src/server/engine_gateway.cc:215-219`), so one overlong request can abort
  every live request.
- **Why it matters:** User-facing requests should get a deterministic 400
  `context_length_exceeded`, not a whole-engine failure. It also means the
  advertised `--max-model-len` option has no effect.
- **Approach:** Wire the resolved limit into `ModelRunner` and
  `AttentionConfig`; check prompt + `max_tokens` (and any retained output) in
  `Scheduler::AddRequest` or the gateway admission path; cap or reject
  `max_tokens`; set prompt-pool `max_prompt_tokens` from the model limit; map
  the error to a 400 in `src/server/api.cc`; add boundary tests at
  `limit-1`, `limit`, and `limit+1`, including chunked prefill and
  preemption replay.

#### P0-3 — HTTP acceptor and sessions lack strand serialization (High)

- **vLLM:** Uses an async serving stack with safe per-connection concurrency;
  connection lifecycle is not shared unsynchronized across worker threads.
- **InferX:** `HttpServer` constructs the acceptor on the raw `io_context`,
  accepts on that raw executor, and posts `Stop()`/`close()` on the same
  raw context (`src/server/http_server.cc:30-65`). `RunServe` runs up to four
  threads on that context (`src/server/serve.cc:173-179`). Beast's stream
  contract and Asio's acceptor contract require an implicit or explicit
  strand for these operations.
- **Evidence:** The executor probe in
  `benchmarks/qwen3/serve_results/rerun_20260923/CONCURRENCY.md` reports
  `socket_has_strand: false`, `coroutine_has_strand: false`, and up to four
  simultaneous callbacks through the stream executor.
- **Why it matters:** This is an executor-contract violation with possible
  races between accept/close, timer/I/O completion, and session teardown,
  especially around draining and disconnects.
- **Approach:** Give the listener one strand; accept each socket onto a
  per-connection strand and spawn the session on that same strand; close and
  cancel through the strand. Do not serialize all connections through one
  global strand. Add a test that asserts strand affinity for accept, close,
  read, write, and timer handlers.

#### P0-4 — Shutdown grace period and engine health are not reliable (High)

- **vLLM:** Graceful shutdown stops admission, cancels/drains work under a
  deadline, and health reports engine failure.
- **InferX shutdown:** The signal handler requests stop, arms a timer that
  only calls `io.stop()`, and later `RunServe` calls blocking
  `gateway.Shutdown()` (`src/server/serve.cc:213-239`). The engine loop exits
  only after `pending` is empty and the scheduler has drained
  (`src/server/engine_gateway.cc:159-165`). Long-running generation can
  therefore outlive the grace period; stopping the I/O context does not stop
  the engine thread, and the final join can block for the remaining workload.
- **InferX health:** `OpenAiHandler::ReadyToServe` checks draining and prompt
  pool health, but the `engine_ready` callback supplied by `RunServe` is only
  `!draining` (`src/server/serve.cc:194-196`). A fatal engine error sets
  `stopping` in the gateway (`src/server/engine_gateway.cc:145-156`) but is
  never reflected in `/health` or `/readyz`. The KV-pressure probe observed
  exactly this: stalled requests with health returning 200.
- **Approach:** Expose a gateway health/fatal state and use it in
  `ReadyToServe`; make the grace deadline actually cancel/abort live engine
  work; separate "stop admission" from "join engine"; perform the join off
  the I/O callback and respect the deadline; add shutdown and fatal-engine
  probes that assert process exit and 503 readiness.

#### P0-5 — Submission can race with engine exit (High)

- **vLLM:** Admission and engine shutdown are linearized so an accepted
  request always has a consumer.
- **InferX:** `EngineGateway::Submit` checks `stopping` before acquiring `mu`,
  then appends to `pending` inside the lock
  (`src/server/engine_gateway.cc:354-378`). An I/O worker can read
  `stopping == false`, pause, then append after the engine has observed an
  empty queue and exited. The submission returns success but no engine
  remains to emit a terminal event.
- **Why it matters:** A caller can wait forever and the in-flight count stays
  reserved. This is especially reachable during draining or after a fatal
  engine error.
- **Approach:** Recheck `stopping` inside the same mutex that guards
  `pending`; have engine exit atomically close admission and drain/fail
  pending submissions; add a deterministic interleaving test and an HTTP
  shutdown-race test.

#### P0-6 — Coroutine/gateway lifetime risk at teardown (Needs Investigation, High)

- **Evidence:** The concurrency investigation records that a suspended
  completion coroutine can hold a cancellation guard that references the
  gateway after gateway destruction
  (`benchmarks/qwen3/serve_results/rerun_20260923/CONCURRENCY.md`, finding 5).
  The audit did not reproduce this with ASan; the report itself labels it an
  ownership inspection rather than a proven failure.
- **Current code:** `RunServe` constructs the gateway before the handler and
  HTTP server and destroys them in reverse (`src/server/serve.cc:181-205`).
  Whether any session/coroutine still references the gateway at teardown
  depends on connection and coroutine cleanup.
- **Why it matters:** Use-after-free at shutdown is a production crash risk.
- **Approach:** Introduce an independently owned cancellation/health state
  whose lifetime exceeds both the gateway and sessions, or explicitly close
  and drain sessions before gateway destruction. Add teardown tests for
  active streams, disconnects, engine fatal errors, and blocked writes.
- **Investigation note:** The older report's claim that encode and decode
  share one mutable tokenizer appears stale for the current tree: `RunServe`
  loads a separate decode tokenizer and each prompt worker owns its own
  tokenizer (`src/server/serve.cc:157-171`). Re-audit the current source
  before acting on that specific finding.

### 4.2 Execution Performance

#### H1 — CUDA graph capture/replay is unimplemented (High)

- **vLLM:** Uses CUDA graphs/compilation as a production decode optimization,
  with graph-safe metadata and preallocated buffers.
- **InferX:** `ModelRunner::CaptureGraphs` explicitly returns
  `UnimplementedError` (`src/engine/model_runner.cc:261-265`). Tensor-parallel
  serving also rejects graphs (`src/engine/model_runner.cc:486-488`). The
  attention backends declare `GraphSupport::kNever`
  (`src/ops/attention/backends/flash_dense.cc:158-170`) and the selector's
  graph environment is never populated (`src/ops/attention/runtime.cc:129-132`).
- **Why it matters:** Per-token decode is launch- and host-overhead sensitive.
  Graphs are one of the largest remaining performance levers, and they are a
  prerequisite for stable high-concurrency serving efficiency.
- **Approach:** Make the decode step graph-safe first: preallocate metadata and
  plan buffers, eliminate per-step host allocation/synchronization, keep
  stable input/output/sampling addresses, add padding and null-slot handling,
  and implement `MetadataBuilder::BuildForCapture`. Then capture uniform
  decode-only batches by size range, replay on shape match, and keep eager
  fallback for prefill/mixed/unsupported geometry. Add graph tests for
  changing batch composition, page turnover, prefix adoption, preemption
  replay, and request finish. Do not advertise graphs for TP until the
  per-rank buffers are stable.

#### H2 — Metadata upload synchronizes the stream repeatedly on the hot path (High)

- **vLLM:** Builds device-resident metadata and uses persistent workspace;
  capture/replay requires no host sync.
- **InferX:** `FillCommonMetadata` enqueues one H2D copy and immediately calls
  `SynchronizeStream` for each array: query offsets, batch indices, positions,
  page offsets, page ids, and last-page lengths
  (`src/ops/attention/fill_common.cc:12-22`, `57-132`). This occurs per cache
  group per step. `FlashDenseKernel::Run` also lazily allocates its plan tensor
  and rebuilds host query-offset vectors on every step
  (`src/ops/attention/backends/flash_dense.cc:106-130`).
- **Why it matters:** This serializes every step, prevents overlap, and is a
  direct blocker for graphs and low TTFT.
- **Approach:** Stage all step metadata into persistent pinned host buffers,
  issue one batched async H2D (or one copy per contiguous range), and use a
  stream event rather than a host synchronize. Preallocate the attention plan
  in `CreateKernel`/runtime initialization. Remove per-step host
  `assign`/`clear`/`push_back` from the kernel path; derive tile counts from
  device metadata or a persistent host mirror.

#### H3 — Every step performs a full stream synchronize and a pageable D2H sample copy (High)

- **vLLM:** Overlaps scheduling/execution where possible and uses pinned
  buffers/events for token handoff.
- **InferX:** After sampling, the runner copies sampled ids into a regular
  `std::vector<int32_t>` and immediately synchronizes the stream
  (`src/engine/model_runner.cc:398-424`). This retires the whole step before
  the scheduler can prepare the next one.
- **Why it matters:** It creates a CPU/GPU bubble each token and makes true
  decode graphs harder. The cost grows with concurrency and model depth.
- **Approach:** Use a persistent pinned host buffer and a recorded event; only
  synchronize when host values are actually required. Evaluate a decoupled
  scheduler/sampler loop or at minimum pipeline the sample D2H with the next
  step's metadata preparation. Measure TTFT/ITL before and after with the
  existing benchmark harness.

#### M7 — Logits and sampling are computed for intermediate prefill chunks (Medium)

- **vLLM:** Only requests that need a next-token distribution incur the LM
  head/sampling work; intermediate chunked-prefill chunks do not produce an
  output token.
- **InferX:** `RunStep` gathers a sample row for every scheduled request and
  calls `ComputeLogits` for the whole batch
  (`src/engine/model_runner.cc:373-384`), then samples every row
  (`:395-420`). The scheduler later discards samples for incomplete prefills
  (`src/engine/scheduler.cc:215-220`).
- **Why it matters:** For long prompts split into many chunks, the LM head is
  paid repeatedly for tokens that are thrown away. It also inflates sampling
  metadata work.
- **Approach:** Distinguish "needs sample" rows from scheduled rows. For a
  prefill chunk that does not complete the sequence, skip `ComputeLogits` and
  sampling entirely; only materialize logits at the final prompt chunk (and
  for decode). Preserve the existing protocol by emitting empty samples for
  skipped rows.

#### M8 — Sampling metadata copies full histories and uploads penalty data every step (Medium)

- **vLLM:** Sampling metadata is packed efficiently and only includes the
  penalty/bias state actually needed.
- **InferX:** For every scheduled request, `RunStep` copies the entire token
  history into a host vector (`src/engine/model_runner.cc:400-413`), and
  `SamplingMetadata::Build` deduplicates and emits history CSR entries even
  when repetition/presence/frequency penalties are disabled
  (`include/inferx/sampling/sampling_metadata.h:127-136`). `UploadParams`
  uploads the resulting entries every step.
- **Why it matters:** Long-context decode repeatedly copies O(context length)
  host data and uploads it, despite the common case using no penalties.
- **Approach:** Skip history construction when `apply_penalties` is false;
  maintain incremental token-count maps per request rather than rebuilding
  from scratch; reuse persistent device buffers; upload only changed CSR
  ranges. Consider a device-side penalty state built once per request.

#### H10 / D3 — MoE execution serializes on the host and has no expert parallelism (Medium)

- **vLLM:** Uses grouped expert GEMMs/token dispatch and optional expert
  parallelism to avoid per-expert host round trips.
- **InferX:** `FusedExperts::Forward` calls `BuildExpertDispatch`, which copies
  expert counts to the host and synchronizes the stream every layer
  (`src/ops/moe.cc:71-86`; `src/models/layers/fused_moe.cc:71-76`). It then
  loops over every expert, issuing dense GEMMs for each occupied slice
  (`src/models/layers/fused_moe.cc:79-104`). All experts are resident on every
  rank; `ParallelConfig::enable_expert_parallel` is not used by the model or
  shard loader.
- **Why it matters:** MoE decode is dominated by dispatch and small GEMMs;
  host synchronization per layer destroys throughput, and the memory footprint
  prevents larger MoE checkpoints.
- **Approach:** Keep routing/dispatch on device, use grouped GEMM or a
  persistent expert kernel, avoid host synchronization, and implement
  EP all-to-all only after a device-side dispatch path exists. Until then,
  reject EP configurations rather than silently ignoring them.

#### M9 — Tensor-parallel logits gather issues one NCCL call per row (Medium)

- **vLLM:** Uses efficient batched collectives and overlaps communication with
  computation.
- **InferX:** `NcclComm::AllGatherLastDim` loops over batch rows and calls
  `ncclAllGather` once per row
  (`src/dist/nccl_comm.cc:183-207`).
- **Why it matters:** With large vocabularies and decode batches, per-row
  collectives add latency and CPU launch overhead.
- **Approach:** Pack row shards into one contiguous buffer and issue one
  all-gather, or use an all-gather-v/reshape path. Keep the existing serial
  loop as a correctness fallback until a batched test passes.

#### M10 — Attention capability checks understate/overstate actual kernel support (Medium)

- **vLLM:** Backend selectors check architecture, dtype, layout, head geometry,
  graph support, and block-size constraints before selecting a kernel.
- **InferX:** `AttentionRuntime::Create` populates only block size and forced
  backend in `Env` (`src/ops/attention/runtime.cc:129-132`), so SM and graph
  requirements are never checked. `flash_dense` declares an empty `head_dims`
  set (meaning "any") while the CUDA implementation only instantiates 64,
  128, and 256 (`src/ops/attention/backends/flash_dense.cc:158-170`,
  `src/ops/cuda/flash_attention.cu:180-189`). Unsupported geometry is rejected
  only at kernel launch.
- **Why it matters:** A model can pass initialization and then fail during
  serving, after weights and cache have been allocated. Graph support and
  architecture selection are also not truthful.
- **Approach:** Populate `Env` from device properties and execution config;
  declare supported head dims/GQA ratios and graph support accurately; add
  startup tests that unsupported geometry is rejected with the capability
  trail. Expose a supported `--attention-backend` option if alternative
  backends are intended.

#### L2 — Native benchmark commands are stubs (Low)

- `src/bench/latency.cc:8` and `src/bench/throughput.cc:8` return
  `UnimplementedError` while the CLI exposes those subcommands. The Python
  benchmark framework is usable, but `inferx bench latency/throughput` does
  not meet its advertised contract. Either implement them using the workload
  harness or remove the commands until they are real.

### 4.3 Scheduling and Batching

#### M4 — `enable_chunked_prefill` is accepted but ignored (Medium)

- **vLLM:** Honors the flag; with it disabled, a prompt that does not fit one
  step is rejected or scheduled under the configured policy.
- **InferX:** The option is parsed into `SchedulerConfig`
  (`include/inferx/config/scheduler_config.h:22-25`,
  `apps/cli/args/engine_args.h:128-135`) but `Scheduler::Schedule` always
  splits by the remaining budget (`src/engine/scheduler.cc:59-65`,
  `121-124`). `long_prefill_token_threshold` is also unused.
- **Why it matters:** Operators cannot reproduce vLLM's non-chunked behavior
  or enforce whole-prompt admission. A configuration value silently changing
  meaning is worse than an explicit rejection.
- **Approach:** When disabled, reject prompts longer than the token budget
  (or the model slot), or schedule them only if the full prompt fits; define
  how this interacts with prefix-cache adoption. Add tests for
  `--no-enable-chunked-prefill`.

#### M5 — Scheduling policy is FCFS only; priority/fairness fields are inert (Medium)

- **vLLM:** Supports FCFS and priority scheduling, with queue/admission
  policies.
- **InferX:** `SchedulerConfig::Policy` includes `kPriority`, `kFair`, and
  `kWeightedFair` (`include/inferx/config/scheduler_config.h:8`), but the
  scheduler implements only FCFS and never reads `policy`, request priority,
  or `long_prefill_token_threshold`. `Request::Priority()` is documented as
  unused (`include/inferx/engine/request.h:110-121`).
- **Why it matters:** Multi-tenant and latency-sensitive workloads cannot
  express priority or fairness, and the public config suggests capabilities
  that do not exist.
- **Approach:** Either implement priority/aging with a deterministic
  comparator and tests, or remove the unused policy variants from the public
  surface until supported. Add queue-wait/priority inversion metrics.

#### M6 — Preemption is recompute-only and has no swap/offload (Medium)

- **vLLM:** Supports recompute and swap/offload strategies.
- **InferX:** `PreemptNewestAfter` releases KV and requeues a victim; replay
  recomputes prompt plus retained output
  (`src/engine/scheduler.cc:243-285`). There is no host swap or
  persistent-cache restore path.
- **Why it matters:** Under memory pressure, recomputation can be much more
  expensive than swapping, especially for long prompts. Combined with P0-1,
  preemption policy is not sufficient to guarantee progress.
- **Approach:** First fix the progress policy. Then add optional host
  swap/restore only with explicit transfer ownership and completion events;
  do not reuse recompute semantics as a substitute.

#### L5 / Config hygiene — Several public configuration fields do not affect execution

- `CacheConfig::cache_dtype` and `calculate_kv_scales` are validated but not
  consumed by model/cache construction
  (`include/inferx/config/cache_config.h:40-47`,
  `src/config/cache_config.cc:23-28`); no FP8 KV path exists.
- `CacheConfig::sliding_window` is not used to derive model retention.
- `prefix_caching_hash_algo` accepts `sha256_cbor`, but `ChainHash` always
  writes SHA-256 (`src/cache/cache_manager.cc:18-38`).
- `ModelConfig::seed` is not passed to the sampler; `SamplingMetadata::Build`
  is called without an engine seed (`src/engine/model_runner.cc:414-415`).
- `--max-model-len` is ignored (P0-2).
- `LoRAConfig` and `SpeculativeConfig` are not fields of `EngineConfig`
  (`include/inferx/server/serve.h:28-34`), so they cannot be enabled from
  serving even if their validators pass.
- **Approach:** For every public field, either implement it or reject it at
  startup with a clear message. Add config-to-engine propagation tests.

### 4.4 KV Cache, PagedAttention, and Prefix Caching

#### K1 — The core paged/prefix-cache implementation is present but limited to GPU-resident full-block reuse

- **Implemented:** byte-denominated paged pool, null block 0, refcounts, LRU
  eviction, prefix hash chain, salt namespace, block-aligned chunking,
  prefix adoption/publication, recompute preemption, and basic stats
  (`include/inferx/cache/`, `src/cache/`; scheduler integration in
  `src/engine/scheduler.cc`).
- **Gap vs vLLM:** No host/CPU offload, disk/persistent cache, disaggregated
  prefill/decode KV transfer, connector framework, or remote KV tier. The
  project has no swap path.
- **Why it matters:** GPU-only prefix reuse helps repeated prefixes but cannot
  extend effective cache capacity or support multi-instance handoff.
- **Approach:** Add a completion-aware block-lease abstraction first (needed
  for any asynchronous transfer), then a pinned-host swap tier, then a
  persistent/remote connector. Keep the current single-GPU path as the
  fallback and make every tier explicit in config.

#### K2 — Recurrent and fork cache work is not drained by the runner (Medium)

- **Evidence:** `CacheManager` can produce `RunnerWork` containing slot resets,
  state copies, and block copy-on-write operations
  (`include/inferx/cache/cache_manager.h:62-72`), but there is no caller of
  `TakeRunnerWork()` anywhere in `src/` or `include/`. `CacheManager::Fork`
  and recurrent slot reset are therefore not wired into execution.
- **vLLM:** Has explicit recurrent/Mamba cache management and state
  checkpointing; generic fork/branching support varies by cache path.
- **Why it matters:** Hybrid/recurrent models and beam-style branching cannot
  be enabled safely even though the cache layer exposes APIs for them.
- **Approach:** Drain `RunnerWork` in `ModelRunner` before execution,
  implement device copies/resets on the step stream, and add tests for fork +
  COW and recurrent-slot reset. Until then, reject fork/recurrent model
  configurations explicitly.

#### K3 — All cache groups must use the same per-layer page size (Medium)

- **vLLM:** Different KV cache groups/layers may have different geometry and
  page sizes, with a coordinator computing a common usable hit.
- **InferX:** `PlanCacheGroups` requires every group's per-layer page bytes to
  match (`src/cache/cache_spec.cc:243-250`), and `KvBlockPool` uses one block
  layout across layers. This blocks mixed dense/MLA/sliding-window
  configurations with different per-token sizes.
- **Why it matters:** It prevents hybrid architectures and some multi-head/
  latent layouts from being served even though the spec types exist.
- **Approach:** Generalize pool allocation to per-group/heterogeneous page
  sizes or a larger common page with internal offsets; make the coordinator
  agree on a usable hit across differing group granularities. Add a test with
  deliberately different group page sizes.

#### K4 — Prefix identity does not cover adapters, multimodal content, or model epochs (Medium)

- **vLLM:** Extra hash keys cover LoRA adapters, multimodal inputs/placeholders,
  prompt embeddings, and model/cache identity.
- **InferX:** The hash chain covers tokens plus an optional salt
  (`src/cache/cache_manager.cc:18-58`, `186-235`). There is no adapter id,
  media identity, model revision, or cache epoch in the key.
- **Why it matters:** Once LoRA or multimodal support is added, different
  effective inputs with identical token ids could incorrectly share KV.
- **Approach:** Add typed, versioned cache identity (model/checkpoint hash,
  adapter id, multimodal span hash, embedding source, positional semantics)
  before enabling those features. Keep salt and identity serialization
  deterministic.

#### K5 — Prefix-cache stats and policy controls are not exposed (Low/Medium)

- **vLLM:** Exposes prefix-cache hit/miss, queried tokens, eviction, and
  queueing metrics through Prometheus; supports configurable cache behavior.
- **InferX:** `SchedulerStats` and `CacheManager::Stats` record hits/queries/
  tokens and occupancy (`include/inferx/engine/scheduler.h:18-38`,
  `include/inferx/cache/cache_manager.h:51-61`) but they are not exported by
  `MetricsService`, which only has HTTP and prompt-pool counters
  (`src/server/metrics.cc`). There is no CLI flag for
  `enable_prefix_caching`, eviction policy, or cache reset.
- **Why it matters:** Operators cannot observe whether prefix caching is
  effective or diagnose index eviction.
- **Approach:** Add engine metrics snapshots (queue depth, KV usage, cached/
  evictable/pinned blocks, prefix queries/hits/tokens, preemptions, recomputed
  tokens) and expose a controlled cache reset. Add a CLI flag for prefix
  caching and document the default.

#### K6 — Startup allocation performs a transient double allocation for heterogeneous pools (Low)

- **Evidence:** `KvBlockPool::Create(...components...)` first creates a uniform
  "probe" pool and then allocates the actual heterogeneous pool before
  replacing the probe (`src/cache/kv_block_pool.cc:67-94`). The peak can be
  close to twice the final size during startup.
- **Why it matters:** A pool that would fit at steady state can OOM during
  initialization.
- **Approach:** Validate dimensions without allocating, compute offsets once,
  then perform one allocation. Add a test where the final pool fits but two
  simultaneous pools do not.

### 4.5 Model, Attention, and Tokenizer Compatibility

#### H8 — Quantization is not supported (Medium)

- **vLLM:** Supports FP8, AWQ/GPTQ and other weight/KV quantization paths,
  including quantized KV cache.
- **InferX:** `BuildModel` rejects any checkpoint with `quantization_config`
  (`src/models/registry.cc:90-95`); linear/attention kernels are BF16 (or
  FP32 on CPU); `ResolveDType` maps only auto/BF16 and rejects FP16/FP32
  execution (`src/models/registry.cc:46-58`). `CacheConfig::cache_dtype` and
  `calculate_kv_scales` are inert (see config hygiene).
- **Why it matters:** Quantized checkpoints are a major part of vLLM's
  deployment envelope and are often necessary for larger models on limited
  GPUs.
- **Approach:** Start with one weight-only format and one KV format behind a
  capability-checked path; add dequant/GEMM kernels, scale handling, and
  numerical tests against BF16. Do not claim FP8 KV until append, attention
  read, scale lifecycle, and sharing/copy paths all work.

#### H9 / M1 — Architecture and attention-mechanism coverage is narrow (Medium)

- **vLLM:** Broad registry of causal, MoE, hybrid, MLA, sliding-window, and
  multimodal models.
- **InferX:** Registry has only Qwen3 dense, Qwen3-MoE, and Llama
  (`src/models/registry.cc:24-31`). Qwen3 and Llama configs reject
  sliding-window attention (`src/models/qwen3/qwen3_config.cc:105-116`,
  `src/models/llama/llama_config.cc:52-56`). MLA, linear/GDN, sparse, chunked
  local, cross-attention, and recurrent cache specs exist, but there is no
  model family or registered kernel backend for MLA/linear/sparse; only dense
  FlashInfer and the dense reference backend are registered
  (`src/ops/attention/selector.cc:9-16`).
- **Why it matters:** The unified attention layer is ahead of model support.
  Features may appear implemented in headers but cannot be exercised by a
  real checkpoint.
- **Approach:** Pick target families (for example Qwen3-Next or
  DeepSeek/MLA) and implement their model adapters plus backend capabilities
  and tests. Until a family is validated, reject it at registry resolution.
  Add a support matrix rather than implying all spec variants are runnable.

#### M3 — Tokenizer and loader formats are narrow (Medium)

- **vLLM:** Loads HF tokenizers including SentencePiece, and many checkpoint
  formats (safetensors, PyTorch, GGUF, sharded/downloaded checkpoints).
- **InferX:** `Tokenizer::FromFile` reads `tokenizer.json` only and explicitly
  rejects `.model` SentencePiece files (`src/tokenizer/tokenizer.cc:92-106`).
  `CheckpointReader::Open` accepts a single `model.safetensors` or a directory
  of `.safetensors` (`src/models/safetensors.cc:33-57`); no PyTorch, GGUF,
  Hugging Face download, or sharded-on-demand loader is wired.
- **Why it matters:** Many Llama/Qwen variants ship SentencePiece or
  non-safetensors assets; users must pre-convert checkpoints and tokenizers.
- **Approach:** Add SentencePiece or a HF-tokenizer compatibility layer; add
  a load-format abstraction and at least PT/GGUF adapters. Preserve the
  existing safetensors path as the validated default.

#### M2 — Multimodal serving is not implemented (Medium)

- **vLLM:** Accepts image/audio/video content parts, preprocesses media, and
  routes model inputs.
- **InferX:** `PromptInput` and `MediaItem` types exist
  (`include/inferx/server/request_model.h:30-116`), but `TextPromptProcessor`
  returns `UnimplementedError("multimodal prompts are not supported yet")`
  (`src/server/prompt_processor_pool.cc:144-145`) and the chat API rejects
  non-string content (`src/server/api.cc:236-244`). `StepInputs` has an
  `input_embeds` path, but no model consumes multimodal embeddings.
- **Why it matters:** This is a major vLLM feature and is already anticipated
  by the request model, so the partial types can mislead callers.
- **Approach:** Implement one modality end-to-end with an explicit media
  processor, model adapter, placeholder/token expansion, prefix-cache
  identity, and streaming response handling. Reject unsupported modalities at
  the API boundary with a stable error.

#### H7 — LoRA and multi-adapter serving are absent (Medium)

- **vLLM:** Loads multiple LoRA adapters, batches requests with different
  adapters, and caches adapters in CPU/GPU memory.
- **InferX:** Only `LoRAConfig` validation exists
  (`include/inferx/config/lora_config.h`, `src/config/lora_config.cc:15-41`).
  There is no adapter loader, linear kernel variant, per-request adapter
  selection, adapter-aware prefix-cache key, or serving API.
- **Why it matters:** Multi-tenant customization is a common vLLM deployment
  requirement.
- **Approach:** Start with one adapter per request and a small resident
  adapter pool; add adapter-aware GEMM or LoRA projection kernels, weight
  loading, per-request validation, and adapter-aware cache identity. Add
  multi-adapter batch tests before exposing an API.

#### H6 — Speculative decoding is design-only (Medium)

- **vLLM:** Has n-gram/prompt-lookup, draft-model, EAGLE/Medusa/MTP and
  rejection-sampling paths.
- **InferX:** `SpeculativeConfig` and method parsing exist
  (`include/inferx/config/speculative_config.h`,
  `src/config/speculative_config.cc`), and `docs/speculative_decode.md`
  describes a multi-phase plan, but the engine has no speculative config
  field, proposer, verifier, or multi-token scheduler commit path. The current
  scheduler explicitly expects one sampled token per scheduled request
  (`src/engine/scheduler.cc:221-236`).
- **Why it matters:** It is a significant decode-latency feature and a clear
  vLLM parity gap, but it depends on reservation/commit semantics and
  multi-row verification.
- **Approach:** Follow the phased plan: add explicit scheduled/computed/
  accepted watermarks and transactional KV reservation, support multiple
  target logits rows, then a greedy n-gram/draft-model chain, then exact
  stochastic verification. Reject enabled speculation until each phase passes
  tests; do not expose the config as if it works today.

### 4.6 Parallelism and Distributed Execution

#### H4 — Only in-process tensor parallelism is wired; other axes are silently ignored (High)

- **vLLM:** Supports tensor, pipeline, data, expert, and context parallelism
  (with combinations and platform-specific restrictions).
- **InferX:** `ParallelConfig` declares DP/PP/CP/TP/EP fields
  (`include/inferx/config/parallel_config.h:35-68`), and `Topology`/`Shard`/
  `ParallelContext` provide mesh and sharding utilities. But
  `ModelRunner::Create` only constructs TP ranks; when building a rank's
  config it creates a fresh `ParallelConfig` containing only tensor size/rank
  (`src/engine/model_runner.cc:510-516`). PP/DP/CP/EP fields are dropped.
  `ParallelConfig::Validate` checks only positivity, not unsupported axes
  (`include/inferx/config/parallel_config.h:129-140`).
- **Why it matters:** A library caller can request PP/DP/EP and receive a
  silently different topology, causing wrong replication/sharding rather than
  an explicit error.
- **Approach:** Add an explicit "supported now" validation that rejects
  `pipeline_parallel_size > 1`, `data_parallel_size > 1`,
  `context_parallel_size > 1`, and `enable_expert_parallel` until each path
  is implemented. Then implement PP stages with point-to-point activations,
  DP replicas/routing, and EP all-to-all in that order. Add topology and
  no-silent-fallback tests.

#### D2 — Tensor parallelism has significant operational limits (Medium)

- Current TP is one process, one GPU per rank, using in-process NCCL
  (`src/engine/model_runner.cc:479-526`). It rejects CUDA graphs
  (`:486-488`) and only BF16 collectives are implemented
  (`include/inferx/dist/comm.h`). There is no multi-node or worker-process
  serving integration: `WorkerPool`/`RunWorker` exist
  (`src/dist/worker_launch.cc`, `src/dist/worker_main.cc`) but no registered
  `inferx worker` command or controller consumes them.
- **Why it matters:** It cannot scale beyond one node/process and cannot use
  the graph optimization. `docs/tensor_parallel.md` correctly describes this
  as an initial serving path, so this is partly an intentional trade-off, but
  it is still a vLLM parity gap.
- **Approach:** Finish the worker-process path and controller, add
  multi-node NCCL/bootstrap, and implement rank-consistent failure handling.
  Only then enable graphs under TP. Keep the current in-process path for
  single-node use.

#### D3 — Expert parallelism is declared but not implemented (Medium)

- `ParallelContext::Moe` and `ShardSpec::kExpert` describe EP partitioning,
  but `FusedExperts` has no EP parameter or all-to-all path
  (`include/inferx/models/layers/fused_moe.h`,
  `src/models/layers/fused_moe.cc`). All experts are replicated on every
  rank.
- **Why it matters:** Large MoE models cannot fit or scale in the way vLLM's
  EP path allows.
- **Approach:** Reject `enable_expert_parallel` until device-side routing,
  all-to-all dispatch/combine, and per-rank expert loading exist. Then add
  tests for token routing across ranks and uneven expert load.

### 4.7 OpenAI API, Sampling, and Observability

#### H5 — Many sampling/API controls are rejected or ignored (High)

- **vLLM:** Supports logprobs, prompt logprobs, n > 1, beam search, logit
  bias, structured/guided decoding, tools, and more sampling controls.
- **InferX internal:** `SamplingParams` already has `logit_bias`,
  `allowed_token_ids`, `logprobs`, `prompt_logprobs`, `min_tokens`, seeds, and
  penalties (`include/inferx/sampling/sampling_params.h`). The GPU sampler
  implements bias/allowlist/penalties/top-k/top-p/min-p in
  `src/sampling/sampling_kernels.cu`; `logprobs` is always absent
  (`include/inferx/sampling/sampler_output.h:18-24`).
- **InferX API:** Explicitly rejects `logprobs`, `logit_bias`,
  `allowed_token_ids`, `prompt_logprobs`, `top_logprobs`, `min_tokens`,
  `response_format`/guided fields, `tools`/`tool_choice`, `n != 1`,
  `best_of`, `echo`, `suffix`, and beam-search fields
  (`src/server/api.cc:187-211`).
- **Why it matters:** Clients expecting vLLM/OpenAI behavior receive 400s for
  common request shapes; structured output and logprobs are especially
  important for agentic and evaluation workloads.
- **Approach:** In order: expose `min_tokens` and `logit_bias`/
  `allowed_token_ids` after fixing their semantics; add logprob computation
  and response schemas; then integrate a grammar/structured-output backend
  (e.g. xgrammar) behind a capability flag; then `n > 1` and beam/branching
  only after state/fork support. Keep explicit `unsupported_parameter` for
  anything not yet implemented.

#### M1 — OpenAI endpoint surface is minimal (Medium)

- **vLLM:** Completions, chat, embeddings/pooling, tokenize/detokenize,
  rerank/score, model list, health/metrics, and tool/reasoning support.
- **InferX:** Only `/health`, `/livez`, `/readyz`, `/version`, `/v1/models`,
  `/metrics`, `/v1/completions`, and `/v1/chat/completions` are routed
  (`src/server/openai_handlers.cc:192-299`). Completions accept one prompt
  string or one token-id array; multiple prompts are not supported
  (`src/server/api.cc:74-86`). Chat supports only string content and the
  Qwen3 template (`src/server/chat_template.cc:14-18`); tools/named messages
  and multimodal content are rejected.
- **Why it matters:** This is a deliberate text-generation-first scope, but it
  is a real compatibility gap for vLLM users and for clients that expect
  tokenization/embedding endpoints.
- **Approach:** Prioritize what the model layer can back: tokenize/
  detokenize, multiple prompts, custom chat templates/tool parsers, then
  embeddings/pooling with a dedicated model path. Add capability-gated route
  registration rather than pretending unsupported routes exist.

#### H12 / O1 — Engine observability is largely absent (Medium)

- **vLLM:** Exposes queueing, KV cache usage, prefix-cache hits, TTFT/ITL/
  TPOT, token counts, preemption, speculative acceptance, and GPU memory via
  Prometheus.
- **InferX:** `MetricsService` records HTTP request counts, prompt-pool
  gauges, disconnects, and aborts (`src/server/metrics.cc:20-49`).
  `SchedulerStats` has running/waiting/KV usage/prefix/preemption counters
  (`include/inferx/engine/scheduler.h:18-38`), but there is no path from the
  engine thread to `/metrics`. There are no latency histograms, per-step
  metrics, prefix-cache gauges, or GPU memory metrics.
- **Why it matters:** Operators cannot diagnose KV pressure, prefix-cache
  effectiveness, request queueing, or throughput regressions from the server.
- **Approach:** Add a thread-safe engine metrics snapshot (counters/gauges
  only, low cardinality), export scheduler/cache stats, add TTFT/ITL histograms
  at the gateway, and expose cache/KV/GPU memory gauges. Keep request ids and
  prompts out of labels.

#### M5 / M6 — Error reporting and request deadlines are coarse (Medium)

- `Request` has an arrival time but no engine-enforced request deadline;
  `GenerateRequest::deadline` exists in the server request model but is not
  propagated to the scheduler/gateway. A slow or stalled request can occupy a
  slot indefinitely.
- Engine failures call `FailAll` and stop the engine
  (`src/server/engine_gateway.cc:145-156`); there is no per-request retry or
  restore path.
- **Approach:** Propagate deadlines into admission and scheduling, expire
  requests with a stable finish reason, and distinguish fatal engine failure
  from request-level failure. Add timeouts to the health/admission path.

#### L2 / O4 — Benchmarking and numerical parity are not current (Low/Medium)

- The C++ latency/throughput commands are stubs (`src/bench/latency.cc:8`,
  `src/bench/throughput.cc:8`) while Python framework benchmarks exist.
- Historical reports show exact greedy agreement failing against vLLM
  (`benchmarks/qwen3/REPORT.md:30-40`), and the later optimization report still
  fails exact equivalence. `docs/bugs.md` records the current explanation as
  BF16 last-ulp/near-tie differences, not a semantic defect.
- **Why it matters:** Performance claims without a current baseline and a
  stated correctness gate are not defensible. The project needs a repeatable
  HEAD benchmark and an explicit numerical acceptance policy.
- **Approach:** Re-run the offline and HTTP suites on the audit HEAD, record
  hardware/software/binary hashes, and compare both throughput and numerical
  agreement. Add a CI smoke test for a fixed checkpoint plus a documented
  tolerance/agreement policy. Do not claim a vLLM win from old artifacts.

## 5. Prioritized TODO List

### P0 — Blocking correctness/liveness/reliability

- [ ] **P0-1: Make the scheduler progress-preserving under KV pressure.**
  Add admission fit checks/headroom, preempt/queue/fail deterministically,
  detect repeated empty plans, and expose a health transition. Test the
  two-request/two-block case and the concurrent HTTP pressure case from
  `benchmarks/qwen3/serve_results/rerun_20260923/CONCURRENCY.md`.
  Refs: `src/engine/scheduler.cc:59-77,126-137`,
  `src/server/engine_gateway.cc:204-219`.
- [ ] **P0-2: Enforce context length and wire `--max-model-len`.**
  Pass the resolved limit into the runner/attention runtime, validate
  prompt + output at admission, cap `max_tokens`, set prompt-pool limits, and
  return a 400 `context_length_exceeded`. Add boundary tests and ensure a
  single bad request cannot fail the whole engine.
  Refs: `include/inferx/config/model_config.h:32-35`,
  `src/engine/model_runner.cc:240-241,429-435`,
  `src/engine/input_batch.cc:29-32,114-115`,
  `src/server/engine_gateway.cc:215-219`.
- [ ] **P0-3: Serialize HTTP accept/session operations with strands.**
  Give the listener and each connection an Asio strand; run accept/close/
  reads/writes/timers on the appropriate strand. Add an executor-affinity
  regression test.
  Refs: `src/server/http_server.cc:30-65`, `src/server/serve.cc:173-179`.
- [ ] **P0-4: Make shutdown enforce its grace period and make health engine-aware.**
  Separate stop-admission from join, cancel live work at the deadline, avoid
  blocking joins on I/O callbacks, and expose gateway fatal/stopping state to
  `/health` and `/readyz`.
  Refs: `src/server/serve.cc:194-196,213-239`,
  `src/server/engine_gateway.cc:145-165`,
  `src/server/openai_handlers.cc:155-160`.
- [ ] **P0-5: Linearize submission against engine shutdown.**
  Recheck `stopping` inside the `pending` mutex; atomically close admission
  and fail/drain pending submissions on engine exit. Add a deterministic
  interleaving test.
  Refs: `src/server/engine_gateway.cc:354-378,159-165`.
- [ ] **P0-6: Audit and fix coroutine/gateway lifetime at teardown (Needs
  Investigation).** Close/drain sessions before destroying the gateway or use
  independently owned cancellation state. Add active-stream, disconnect,
  fatal-engine, and blocked-write teardown tests.
  Ref: `benchmarks/qwen3/serve_results/rerun_20260923/CONCURRENCY.md`
  finding 5; `src/server/serve.cc:181-245`.

### P1 — High-impact performance and compatibility foundations

- [ ] **H1: Implement graph-safe decode CUDA graphs.**
  Preallocate stable metadata/plans/outputs, implement
  `BuildForCapture`, remove hot-path sync/allocation, capture uniform decode
  batches, and replay with eager fallback. Add graph tests for changing
  batches, page turnover, prefix adoption, preemption, and finish.
  Refs: `src/engine/model_runner.cc:261-265,486-488`,
  `include/inferx/ops/attention/builder.h`,
  `src/ops/attention/fill_common.cc`,
  `src/ops/attention/backends/flash_dense.cc:106-130`.
- [ ] **H2: Remove synchronous metadata uploads and hot-path allocations.**
  Use persistent pinned staging, batched async H2D, event-based ordering, and
  preallocated attention plans. Remove per-step host vector growth from
  `flash_dense`.
  Refs: `src/ops/attention/fill_common.cc:12-22,57-132`,
  `src/ops/attention/backends/flash_dense.cc:106-130`.
- [ ] **H3: Pipeline token handoff and remove the unconditional step sync.**
  Use a pinned sample buffer plus an event; synchronize only when host values
  are needed. Measure TTFT/ITL before and after.
  Ref: `src/engine/model_runner.cc:398-424`.
- [ ] **H4: Reject or implement non-TP parallel axes.**
  Add explicit startup rejection for PP/DP/CP/EP while they are unsupported;
  do not silently drop them when constructing rank configs.
  Refs: `include/inferx/config/parallel_config.h:35-140`,
  `src/engine/model_runner.cc:510-516`.
- [ ] **H5: Close the sampling/API control gap in a staged order.**
  Add `min_tokens`, logit bias/allowed ids, logprobs, then structured output
  (grammar backend), then `n > 1`/beam-style branching. Keep unsupported
  fields as explicit 400s until each path is implemented end to end.
  Refs: `include/inferx/sampling/sampling_params.h`,
  `src/server/api.cc:187-211`,
  `include/inferx/sampling/sampler_output.h:18-24`.
- [ ] **H12: Export engine observability.**
  Snapshot scheduler/cache stats into `/metrics`: queue depth, running/waiting,
  KV usage, cached/evictable/pinned blocks, prefix hits/queries/tokens,
  preemptions, recomputed tokens, TTFT/ITL histograms, and GPU memory.
  Refs: `include/inferx/engine/scheduler.h:18-38`,
  `include/inferx/cache/cache_manager.h:51-61`,
  `src/server/metrics.cc`.

### P2 — Feature breadth and throughput follow-ups

- [ ] **M4: Honor `enable_chunked_prefill` and `long_prefill_token_threshold`.**
  Define non-chunked admission and reject prompts that cannot fit; add tests.
  Refs: `include/inferx/config/scheduler_config.h:22-29`,
  `src/engine/scheduler.cc:59-65,121-124`.
- [ ] **M5: Implement scheduling policies or remove the unused policy enum.**
  Add priority/aging/fairness with deterministic comparison and starvation
  tests, or hide unsupported policies from the public config.
  Refs: `include/inferx/config/scheduler_config.h:8,27`,
  `src/engine/request_queue.h`.
- [ ] **M6: Add an explicit preemption/offload policy.**
  First fix P0-1; then add optional host swap with completion-aware block
  leases. Do not rely on recompute for all pressure cases.
- [ ] **M7: Skip logits and sampling for intermediate prefill chunks.**
  Materialize logits only for rows that need a token; emit empty samples for
  skipped rows. Refs: `src/engine/model_runner.cc:373-420`,
  `src/engine/scheduler.cc:215-220`.
- [ ] **M8: Make sampling metadata incremental.**
  Skip history CSR when penalties are disabled; maintain per-request counts;
  reuse device buffers; upload only deltas.
  Refs: `include/inferx/sampling/sampling_metadata.h:127-136`,
  `src/engine/model_runner.cc:400-420`.
- [ ] **M9: Batch tensor-parallel logits collectives.**
  Replace per-row `ncclAllGather` with one packed collective; keep the serial
  path as a fallback. Ref: `src/dist/nccl_comm.cc:183-207`.
- [ ] **M10: Make attention capability checks truthful.**
  Populate `Env` SM/graph requirements, declare supported head dims/GQA ratios,
  and reject unsupported geometry at initialization. Add capability tests and
  expose a supported attention-backend selector if multiple backends remain.
  Refs: `src/ops/attention/runtime.cc:129-132`,
  `src/ops/attention/backends/flash_dense.cc:158-170`,
  `src/ops/cuda/flash_attention.cu:180-189`.
- [ ] **H10/D3: Build device-side MoE dispatch and expert parallelism.**
  Remove the per-layer host synchronization, use grouped/persistent expert
  kernels, then add EP all-to-all and reject the config until ready.
  Refs: `src/ops/moe.cc:71-86`,
  `src/models/layers/fused_moe.cc:71-104`,
  `include/inferx/config/parallel_config.h:60-68`.
- [ ] **H11: Add KV offload/tiering after a block-lease foundation.**
  Implement pinned-host swap first, then persistent/remote tiers and
  disaggregated KV handoff. Reject transfers until ownership and completion
  semantics are explicit.
- [ ] **K2: Drain `CacheManager::RunnerWork` in the runner.**
  Implement block COW, slot reset, and recurrent-state copy on the step stream;
  add fork/recurrent tests. Refs:
  `include/inferx/cache/cache_manager.h:62-72`,
  `src/cache/cache_manager.cc:292-293,361-386`.
- [ ] **K3/K4: Generalize cache page/group identity.**
  Support heterogeneous group page sizes and include adapter/media/model
  identity in cache keys before enabling those features.
  Refs: `src/cache/cache_spec.cc:243-250`,
  `src/cache/cache_manager.cc:18-58,186-235`.
- [ ] **H8: Add quantization support.**
  Start with one weight-only format and one KV format, with scales and
  numerical tests. Reject quantized checkpoints until the path is complete.
  Refs: `src/models/registry.cc:90-95`,
  `src/models/registry.cc:46-58`,
  `include/inferx/config/cache_config.h:40-47`.
- [ ] **H7: Add single-adapter then multi-adapter LoRA.**
  Implement adapter loading, projection kernels, per-request selection,
  adapter-aware cache identity, and serving API. Refs:
  `include/inferx/config/lora_config.h`, `src/config/lora_config.cc:15-41`.
- [ ] **H6: Implement speculative decoding in phases.**
  Transactional KV reservation/commit, multi-row target logits, greedy n-gram/
  draft chain, then exact stochastic verification. Keep it disabled until each
  gate passes. Refs: `docs/speculative_decode.md`,
  `src/config/speculative_config.cc`, `src/engine/scheduler.cc:221-236`.
- [ ] **H9: Expand model/attention families deliberately.**
  Add target families with kernel backends and tests; reject unvalidated
  families at registry resolution. Refs:
  `src/models/registry.cc:24-31`,
  `src/models/qwen3/qwen3_config.cc:105-116`,
  `src/models/llama/llama_config.cc:52-56`,
  `src/ops/attention/selector.cc:9-16`.
- [ ] **M2: Add one multimodal path end to end.**
  Implement media preprocessing, model adapter, placeholder expansion,
  cache identity, and API/rejection behavior. Refs:
  `src/server/prompt_processor_pool.cc:144-145`,
  `src/server/api.cc:236-244`,
  `src/server/chat_template.cc:30-38`.
- [ ] **M3: Add SentencePiece/HF-compatible tokenizer and loader formats.**
  At minimum support SentencePiece and PyTorch/GGUF loading behind explicit
  formats. Refs: `src/tokenizer/tokenizer.cc:92-106`,
  `src/models/safetensors.cc:33-57`.
- [ ] **M1: Expand the OpenAI-compatible surface carefully.**
  Tokenize/detokenize, multiple prompts, custom chat templates/tool parsers,
  then embeddings/pooling and multi-model routing. Keep capability-gated
  routes explicit. Ref: `src/server/openai_handlers.cc:192-299`.

### P3 — Ergonomics, cleanup, and validation

- [ ] **L1: Wire or remove the worker IPC/launcher.** `WorkerPool`/`RunWorker`
  exist but no CLI command or controller uses them. Either integrate them for
  multi-process TP/PP or mark them internal and unshipped.
  Refs: `src/dist/worker_launch.cc`, `src/dist/worker_main.cc`,
  `apps/cli/commands.h`.
- [ ] **L2: Implement or remove `bench latency`/`bench throughput`.**
  Refs: `src/bench/latency.cc:8`, `src/bench/throughput.cc:8`.
- [ ] **L3: Implement or remove stub CLI commands.** `chat`, `complete`,
  `run-batch`, `collect-env`, and `launch render` currently return
  Unimplemented.
- [ ] **L4: Make CPU device handling truthful.** Either add a CPU model
  execution path or reject `--device cpu` for model serving at CLI/config
  validation instead of failing in `Worker::InitDevice`.
  Refs: `apps/cli/args/model_config_args.h:50-61`,
  `src/engine/worker.cc:37-42`.
- [ ] **L5: Implement or reject every public config field.**
  `cache_dtype`, `calculate_kv_scales`, `sliding_window`,
  `prefix_caching_hash_algo`, `max_model_len`, `seed`, `LoRAConfig`, and
  `SpeculativeConfig` currently have partial/no propagation. Add startup
  validation and propagation tests.
- [ ] **O4: Establish a current benchmark and numerical-parity gate.**
  Freeze the audit HEAD, re-run offline and HTTP benchmarks, record hashes,
  compare against vLLM 0.29.0, and document the numerical acceptance policy.
  Refs: `benchmarks/qwen3/REPORT.md`, `benchmarks/qwen3/OPTIMIZATION_20260922.md`,
  `benchmarks/qwen3/serve_results/rerun_20260923/`.
- [ ] **O1: Add a cache diagnostic surface.** Expose occupancy breakdown,
  eviction count, hit tokens by group, and an optional invariant checker.
  Refs: `src/cache/kv_block_pool.cc`, `src/cache/cache_manager.cc`.

## 6. Recommended Roadmap

The ordering below follows dependencies, not feature visibility. Do not start
graph or speculative work before the step contract is stable.

### Phase 0 — Baseline and guardrails (days)

1. Freeze the current HEAD and record build/test/benchmark provenance.
2. Add regression tests for the known P0 cases: KV-pressure livelock,
   context-length failure, shutdown race, submit-after-stop, and HTTP strand
   affinity.
3. Add config propagation tests so no public field silently does nothing.

### Phase 1 — Correctness and serving liveness (1–2 weeks)

1. P0-1 scheduler progress and P0-2 context limits.
2. P0-3 HTTP strands, P0-4 shutdown/health, P0-5 submit/stop linearization,
   P0-6 lifetime audit.
3. Add engine health metrics and make `/health`/`/readyz` truthful.
4. Re-run the concurrency probes and the full HTTP matrix; require zero
   stalls and bounded shutdown.

### Phase 2 — Performance foundations (2–4 weeks)

1. H2 metadata staging and H3 token handoff; remove per-step sync/allocation.
2. H1 stable decode buffers and CUDA graph capture/replay.
3. M7/M8 prefill/sampling work elimination.
4. Re-benchmark offline and HTTP, including TTFT/ITL and graph variants.

### Phase 3 — Scheduling and cache parity (2–4 weeks)

1. M4/M5/M6 configurable scheduling, chunked-prefill semantics, and
   preemption/offload policy.
2. K2 runner work, K3 heterogeneous groups, K4 cache identity.
3. H11 host swap/tiering only after completion-aware block leases.
4. M10 capability correctness and H10 MoE dispatch.

### Phase 4 — Feature breadth (parallelizable after Phase 2)

1. H5 sampling/API controls, then structured outputs and logprobs.
2. H7 LoRA, H8 quantization, H6 speculative decoding in their own
   capability-gated tracks.
3. H9/M2/M3 model, multimodal, and tokenizer/loader coverage.
4. M1 additional API endpoints only when backed by a real model path.

### Phase 5 — Scale and operations

1. H4/D2 pipeline/data/expert/context parallelism and worker-process
   multi-node execution.
2. KV offload/remote transfer and optional disaggregated prefill/decode.
3. Full Prometheus dashboard, SLO tests, and continuous performance
   regression gates.

### Dependency notes

- P0-1 and P0-2 should land before graph/performance work: graph replay
  amplifies state/accounting bugs.
- H2 and H3 are prerequisites for H1.
- H6 depends on transactional KV reservation/commit, which is also useful for
  branching and preemption.
- H7/H8/M2/K4 share cache identity work; do not enable them without K4.
- H4 cannot be tested meaningfully without the worker-process path and
  topology integration; reject unsupported axes until then.
