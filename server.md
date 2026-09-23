# InferX Serving Layer — API Definition and Implementation Plan

Status: proposal (no code written yet). Scope: `inferx serve`, an
OpenAI-compatible HTTP server whose first-class client is
`vllm bench serve`, benchmarked against vLLM on Qwen3-0.6B (BF16, single GPU).

Everything in the "vLLM bench contract" section is derived from the installed
benchmark source (`python/.venv/.../vllm/benchmarks/serve.py`,
`lib/endpoint_request_func.py`, `lib/ready_checker.py`,
`datasets/datasets.py`), not from memory; the installed source stays the
authority and the version is recorded per run, as `run.py` already does.

## 1. Requirements

1. `vllm bench serve --backend openai` must work against `inferx serve`
   unmodified: same endpoint path, same request fields, same SSE parsing
   rules, same success criteria.
2. The server must measure honestly: per-token flush, no buffering that would
   distort TTFT/ITL, admission and abort behavior that keeps the engine busy
   without unbounded queueing (plan.md §(h)).
3. The engine stays single-threaded by design (`scheduler.h`): the scheduler
   and model runner are stepped by exactly one thread; the server is a
   front-end around that loop, never a second driver.
4. Keep the existing CLI contract: `serve` flags (`--host`, `--port`,
   `--served-model-name`, Engine group, Sampling group) are already
   registered; `ServeParams` (`include/inferx/server/serve.h`) already
   carries them.
5. Non-goals for this milestone: `/v1/chat/completions`, TLS, multi-model,
   distributed serving, Prometheus `/metrics`, logprobs, `n > 1`, token-array
   prompts, request prioritization. All are additive later.

## 2. vLLM bench serve contract (grounded in installed source)

**Endpoint.** `--backend openai` posts to `--api
http://HOST:PORT/v1/completions` (the URL must end in `completions`).
Requests are `application/json`; `Authorization: Bearer …` is sent only if
`OPENAI_API_KEY` is set; `x-request-id` may be set. Both must be accepted and
ignored.

**Request body** (exactly what the bench sends):

```json
{
  "model": "<name>",
  "prompt": "<text>",
  "repetition_penalty": 1.0,
  "max_tokens": 256,
  "logprobs": null,
  "stream": true,
  "stream_options": {"include_usage": true},
  "ignore_eos": true
}
```

Plus, when `--temperature/--top-p/--top-k/--min-p/--frequency-penalty/
--presence-penalty/--repetition-penalty` are passed, those fields are merged
in. Notes that shape the design:

- `repetition_penalty: 1.0` is **always** sent — the server must accept the
  field. Neutral values (1.0 / 0) must be accepted and treated as no-ops.
- `logprobs` is sent as `null` — accept `null`; any non-null value is
  rejected with 400 for now.
- `temperature` is **no longer defaulted to 0 by the bench** — the server's
  default applies. For deterministic comparisons we always run the bench with
  `--temperature=0` and `--ignore-eos` (the bench also auto-sets
  `ignore_eos=true` for openai-compatible backends with fixed output lens).
- Prompts are **text**. The random dataset generates token ids, decodes them
  to text, and sends text; tokenization therefore happens inside the
  measured path on *both* engines — symmetric, fair, and it means
  `prompt_token_ids` support is not required for parity.

**Streaming response parsing** (what the bench accepts):

- Any non-200 status marks the request failed (`response.reason` recorded).
- Body is consumed as a byte stream; SSE messages are framed, `data: ` is
  stripped, lines starting with `:` (comments/pings) are skipped, and
  `data: [DONE]` terminates the stream.
- Every chunk carrying `choices` updates timestamps: the first such chunk is
  TTFT, subsequent ones are ITL samples. `choices[0].text` may be `null` or
  `""` (special tokens) and still counts as an event.
- A chunk **without** `choices` but with `usage` supplies
  `usage.completion_tokens` (required for token accounting) and optionally
  `usage.prompt_tokens`. Success requires at least one choices-bearing chunk.
- => The server must emit, in order: token chunks with `choices[0].text`,
  then a final usage-only chunk (`choices: []`, per OpenAI
  `include_usage` semantics), then `data: [DONE]`.

**Readiness and warmup.** The bench does not poll `/health`: it issues a real
streaming test request and retries every 5 s for up to 600 s, then runs
`--num-warmups` warmup requests before the timed phase. The server only needs
to answer correctly; slow startup is tolerated. `/v1/models` is fetched only
when `--model` is omitted, to auto-detect the model name.

**Metrics** are computed client-side (TTFT/ITL/throughput percentiles). The
server contributes nothing but honest timing.

## 3. API design

### 3.1 Endpoints

| Method | Path                | Phase | Purpose                                  |
|--------|---------------------|-------|------------------------------------------|
| POST   | `/v1/completions`   | 1     | Text completions, streaming + non-streaming |
| GET    | `/v1/models`        | 1     | Model discovery (`served-model-name`)    |
| GET    | `/health`           | 1     | Liveness: 200 once the engine is up      |
| POST   | `/v1/chat/completions` | 2+ | Chat template rendering                  |
| GET    | `/metrics`          | later | Prometheus text format                   |

Unknown paths → 404 with a JSON error body.

### 3.2 `POST /v1/completions`

Request fields (unknown fields are ignored for forward compatibility, as
vLLM's protocol layer does):

| Field | Type | Handling |
|---|---|---|
| `model` | string | Must equal `served-model-name` (or the model dir basename); else 404. |
| `prompt` | string | Required. Encoded with the checkpoint tokenizer. |
| `max_tokens` | int ≥1 | Maps to `SamplingParams.max_tokens`. Required in practice (bench always sends). |
| `temperature` | float ≥0 | `SamplingParams.temperature`; server default from `serve`'s Sampling group. |
| `top_p` | float (0,1] | `SamplingParams.top_p`. |
| `ignore_eos` | bool | `SamplingParams.ignore_eos`. |
| `stream` | bool | Default false; bench sends true. |
| `stream_options.include_usage` | bool | Accept; usage chunk is always emitted (it is harmless when unrequested). |
| `repetition_penalty`, `frequency_penalty`, `presence_penalty` | number | Passed through into `SamplingParams` (which carries them) and range-validated by `SamplingParams::Validate()`; invalid values → 400. |
| `logprobs` | null / int | `null` accepted; non-null → 400. |
| `seed` | int | Passed through to `SamplingParams.seed`. |
| `top_k`, `min_p` | number | Passed through into `SamplingParams` with the same validation. |

Validation errors return 400 with the OpenAI error object naming the field.

**Streaming response** (`200`, `Content-Type: text/event-stream`,
chunked):

```
data: {"id":"cmpl-<reqid>","object":"text_completion_chunk","created":<unix>,
       "model":"<name>","choices":[{"index":0,"text":" token","finish_reason":null}]}

data: {...same..., "choices":[{"index":0,"text":"","finish_reason":"length"}]}

data: {"id":"cmpl-<reqid>","object":"text_completion_chunk","created":<unix>,
       "model":"<name>","choices":[],
       "usage":{"prompt_tokens":128,"completion_tokens":256,"total_tokens":384}}

data: [DONE]

```

- One SSE event per generated token, flushed immediately (`text` carries the
  incremental detokenized delta, possibly empty for special tokens — the
  bench tolerates empty text but still timestamps it).
- `finish_reason`: `"stop"` (EOS/stop) or `"length"` (token cap) on the last
  token-bearing chunk; `null` before.
- Optional keep-alive comment lines (`: ping`) during long prefills; the
  bench skips them.

**Non-streaming response**: standard completion object with full `text`,
`finish_reason`, and `usage` (needed for `inferx complete` and curl; the
bench does not use it).

**Errors**: HTTP status + OpenAI error body
`{"error":{"message","type","param","code"}}`. Mapping: malformed
JSON/invalid fields → 400; unknown model → 404; engine queue full → 503 with
`Retry-After` (bounded admission below); unsupported feature values → 400;
engine failure mid-stream → emit a final SSE chunk with an error object then
`[DONE]` (the bench will mark the request failed — correct behavior), and log
server-side.

### 3.3 `GET /v1/models`

`{"object":"list","data":[{"id":"<served-model-name>","object":"model",
"created":<unix>,"owned_by":"inferx"}]}` — one entry. This is what the bench
hits when `--model` is omitted.

### 3.4 `GET /health`

200 with empty body once the model runner is initialized and the engine loop
is running; 503 before. Cheap (no engine interaction) so it never distorts
timings.

## 4. Server architecture

```
            io_context (M threads, coroutines only)
  ┌──────────────────────────────────────────────────────┐
  │ Listener (accept loop)                               │
  │   └─ HttpSession (1 coroutine per connection)        │
  │        ├─ parse HTTP (beast)                         │
  │        ├─ Route: /v1/completions → CompletionsHandler│
  │        │    ├─ validate JSON → RequestSpec           │
  │        │    ├─ Tokenizer::Encode (I/O thread)        │
  │        │    └─ EngineGateway::Submit ───────────┐    │
  │        │         └─ StreamHandle (asio channel) │    │
  │        │    StreamingResponder (SSE writer      │    │
  │        │         coroutine; co_awaits channel)  │    │
  │        └─ /v1/models, /health (static)          │    │
  └─────────────────────────────────────────────────┼────┘
                                                     │ events (posted)
  ┌──────────────────────────────────────────────────┼────┐
  │ Engine thread (exactly one; owns CUDA stream)    │    │
  │   loop: Scheduler::Schedule → ModelRunner::Run   │    │
  │         → UpdateFromOutput → PopFinished         │    │
  │   for each new token: incremental detokenize,    │    │
  │         net::post(io, channel.try_send(event))   │◄───┘
  │   admission: AddRequest; ResourceExhausted → 503 │
  │   aborts: AbortRequests on cancel/disconnect     │
  └───────────────────────────────────────────────────┘
```

Components live in `src/server/` behind `include/inferx/server/`:

- **`http_server.{h,cc}`** — Listener + HttpSession: accept, read, route,
  write; knows nothing about the engine.
- **`api.{h,cc}`** — request/response JSON schemas, validation, OpenAI error
  objects (nlohmann JSON, already vendored).
- **`engine_gateway.{h,cc}`** — the only engine-aware piece: owns the engine
  thread, exposes `Submit/Cancel/Stats`, owns the id map
  (string `cmpl-…` ↔ `RequestId`).
- **`detokenizer.{h,cc}`** — incremental decode state per request.
- **`serve.cc`** — assembles `ServeParams` → components; runs until shutdown.

**Dependency prerequisite.** `third_party/beast` (361) vendors only
`boost/beast`; Beast requires Boost.Asio, and no Boost exists on the machine
or in the tree. Delivered: a checked-in, SHA-pinned header-only subset under
`third_party/boost` (provenance in `third_party/boost/fetch.sh`). Note the
version pairing: Beast 361 is a post-1.89 develop snapshot whose
`core/detail/static_assert.hpp` dependency does not exist in the 1.89
release, so the subset is pinned to develop commit SHAs (recorded in
`third_party/boost/SHAS.txt`), not release tags. No compiled Boost libraries
are needed. CMake exposes `inferx::boost` / `inferx::beast` interface
targets from `cmake/InferXDependencies.cmake`. The coroutine path is proven
by Beast's own `example/http/server/awaitable` (used as the compile gate).

### Delivery notes

The dependency subset is vendored and the compile gate (Beast's awaitable
example) passes. SSE bodies use manual HTTP chunked framing
(`hex-size CRLF data CRLF` via `net::async_write`): Beast's `chunk_body`
is a buffer sequence, not a writable message, and manual framing keeps
per-event writes flush-tight. Delta events are coalesced into one write per
engine step (draining already-queued channel events), so syscalls scale with
steps, not tokens. The finish reason rides on the last content chunk (one
delta is held back), matching OpenAI streams without distorting ITL. The
incremental detokenizer lives inside `engine_gateway.cc` (no separate
`detokenizer.{h,cc}`): phase-1 full re-decode per step with suffix emission,
as planned in §7.

## 5. Coroutine model

- **One coroutine per connection** (`asio::awaitable<void>`), spawned with
  `co_spawn(io_context, …, detached)` from the Listener's accept loop. All
  Beast calls use `asio::use_awaitable`. Keep-alive: the session loops
  reading requests until EOF/error.
- **Streaming = a second awaitable chain per request**: after headers are
  written, the session (or a child coroutine) loops
  `co_await channel->async_receive()` → build SSE event → `async_write`,
  flushing one event per token. No timers in the hot path; TTFT/ITL fidelity
  comes from writing immediately per event.
- **Engine → I/O handoff.** The engine thread never touches asio objects
  directly; it `net::post`s a completion onto `io_context` that
  `try_send`s the event into the per-request
  `asio::experimental::channel`. Channels give natural backpressure shape
  but must not block the engine thread: bounded per-request pending buffer
  (default 4 MiB of text). On overflow (a stalled consumer), the gateway
  aborts the engine request and emits an error finish event — mirroring
  vLLM's slow-client handling, protecting the measured requests.
- **Cancellation.** Client disconnects surface as write errors in the
  streaming coroutine; the session calls `EngineGateway::Cancel`, which
  enqueues the id for `Scheduler::AbortRequests` on the next engine
  iteration. The engine never blocks on I/O; I/O never blocks on the engine.
- **Timeouts.** `beast::tcp_stream` with `expires_after`: 30 s to receive a
  complete request head; keep-alive idle timeout 120 s; **no** write timeout
  while streaming (generation time is legitimate; liveness is detected via
  aborts/disconnects).
- **Shutdown.** SIGINT/SIGTERM → stop accepting, reply 503 to new requests,
  stop the engine loop after in-flight requests drain (grace 10 s, then
  abort remaining), run `io_context.stop()`, join threads. Clean shutdown
  keeps benchmark automation simple.

## 6. Request lifecycle

```
HTTP request
  → parse+validate (400 on error) ───────────────► error object, done
  → model check (404) 
  → Tokenizer::Encode(prompt) → token ids (400 on failure)
  → EngineGateway::Submit(RequestSpec)
       engine: Scheduler::AddRequest
         ResourceExhausted (queue_capacity) ─────► 503 + Retry-After
       queued → engine loop admits (budget/KV) → running
  → first token event → first SSE chunk (TTFT)
  → per token: incremental detokenize delta → SSE chunk (ITL sample)
  → finish event (kStopped→"stop" | kLengthCapped→"length")
       → final token chunk with finish_reason
       → usage chunk {prompt_tokens, completion_tokens, total_tokens}
       → data: [DONE]
  → connection returns to keep-alive pool

Failure/abort paths:
  • client disconnect at any point → write error → Cancel → AbortRequests
    → KV freed; nothing emitted.
  • engine error for this request → error chunk + [DONE], request resources
    freed; other requests unaffected.
  • slow consumer past pending cap → abort as above, error chunk.
```

Usage accounting is exact: `prompt_tokens = prompt.size()`,
`completion_tokens = output.size()` from the finished `Request` — the same
numbers `analyze.py`-style validation can cross-check.

## 7. Concurrency strategy

- **I/O threads:** `M = clamp(hardware_concurrency, 2, 4)` threads running
  one `io_context`. Sessions are coroutines — thousands are cheap; M only
  needs to cover JSON parse and SSE write costs. Pin via a `--io-threads`
  flag later if needed.
- **Engine thread:** exactly one, owning scheduler, runner, and CUDA stream
  (the scheduler's documented single-threaded design). All handoffs are
  value-type events; no engine state is shared with I/O threads except
  through the gateway's mutex-guarded submit/abort lists (tiny, contended
  only at request boundaries, never per token).
- **Admission control:** `SchedulerConfig.queue_capacity` (already exists,
  default 64) bounds waiting requests; overflow → 503 + `Retry-After: 1`.
  `max_num_seqs` / `max_num_batched_tokens` continue to bound the running
  set — the same knobs the benchmark pins on both engines.
- **Detokenization:** on the engine thread (cheap, preserves ordering, keeps
  I/O threads free). Phase 1: re-decode the request's accumulated output per
  step (O(n²) overall, negligible at ≤2048-token contexts for 0.6B-scale
  runs). Phase 2 if profiling shows cost: windowed incremental decode with
  one-token readahead, as in vLLM's detokenizer.
- **Fairness:** FCFS scheduling (the scheduler's policy); the bench's
  arrival shaping (Poisson via `--request-rate` or closed-loop bursts) is
  client-side and needs no server support.

## 8. Interfaces to implement

```cpp
// engine_gateway.h — the seam between server and engine.
struct RequestSpec {
  std::vector<TokenId> prompt;      // encoded by the caller
  sampling::SamplingParams params;  // temperature, top_p, max_tokens, ignore_eos
};

struct StreamEvent {                 // value type crossing the seam
  enum class Kind { kToken, kFinish, kError } kind;
  std::string delta;                 // detokenized text delta (kToken)
  FinishReason reason;               // kFinish
  int prompt_tokens, completion_tokens;  // kFinish (usage chunk)
  Status error;                      // kError
};

class EngineGateway {
 public:
  // Validates capacity; never blocks. Sends to the engine via a
  // mutex-guarded pending list drained at the next loop iteration.
  StatusOr<StreamHandle> Submit(RequestSpec spec);
  void Cancel(RequestId id);         // async; applied next iteration
  void Drain(std::chrono::steady_clock::time_point deadline);  // shutdown
};

class StreamHandle {
 public:
  // Bounded channel of StreamEvent; consumed by the SSE writer coroutine.
  asio::experimental::channel<void(StatusOr<StreamEvent>)>& events();
  RequestId id() const;
};
```

Engine-side additions (small, contained):

- `FinishReason` already exists and maps 1:1 (`kStopped→"stop"`,
  `kLengthCapped→"length"`); `kAborted/kError` are internal-only.
- Engine loop module (`src/server/engine_loop.cc`): the synchronous
  Schedule→Run→UpdateFromOutput→PopFinished cycle `bench workload` already
  performs, lifted into a reusable loop with a per-iteration callback hook.
  No scheduler or runner changes required.
- `Tokenizer::Encode/Decode` already virtual and file-loadable; the
  detokenizer wraps `Decode`.
- `ServeParams` needs no new fields for phase 1 (host, port,
  `served-model-name`, runner, scheduler, default sampling are present).

Sampling-field policy (penalties/`top_k`/`min_p` neutral-accept, non-neutral
400) keeps `SamplingParams` — which has exactly temperature/top_p/max_tokens/
ignore_eos — sufficient until those samplers exist.

## 9. Testing

- **Unit (`tests/server_*_test.cc`, no GPU needed for HTTP paths):** SSE
  framing (multi-event writes, `data:`/`[DONE]`, comment lines), JSON schema
  validation matrix (every field: missing/invalid/neutral/non-neutral),
  error-object shapes and status codes, channel backpressure and
  slow-consumer abort, id-map lifecycle.
- **Integration:** fake gateway emitting deterministic events → assert
  byte-exact SSE bodies via a local socket; `curl` recipes in the README.
- **Bench parity (GPU, GPU-lock discipline as in `run.py`):**
  1. capture-and-replay the bench's exact requests against `inferx serve`
     (python asyncio client that mirrors `endpoint_request_func.py`'s
     parser), asserting success and usage accounting;
  2. full `vllm bench serve` runs against both engines, serially, same GPU,
     with pinned conditions (below);
  3. extend `benchmarks/qwen3/analyze.py` to compare bench-serve JSON
     outputs across engines.
- **Pinned fairness conditions (both sides):** `models/Qwen3-0.6B` BF16;
  `--temperature 0 --ignore-eos`; same `--max-num-seqs`, token budget 4096,
  block size 16, KV bytes
  (`--kv-cache-memory-bytes 3758096384` ↔ `--num-kv-blocks 2048`); graphs
  paired (`--cuda-graphs` ↔ vLLM default); prefix caching off; chunked
  prefill on; `VLLM_USE_V2_MODEL_RUNNER=0` on WSL; identical dataset and
  seed (`--dataset-name random --random-seed N`, fixed lengths).

## 10. Milestones

1. **Deps:** vendor the Boost.Asio subset; build Beast's awaitable example
   as a compile gate; wire into CMake as `inferx::http`.
2. **Skeleton:** Listener + HttpSession + routing + `/health` + `/v1/models`
   + error objects; unit tests.
3. **Engine loop + gateway:** extract the workload loop; gateway with
   channels, submit/cancel/abort; fake-event integration test.
4. **Completions, non-streaming:** full request validation, encode, run,
   aggregate response; `inferx complete` gets a real server to talk to.
5. **Streaming:** incremental detokenizer + SSE writer + usage chunk +
   `[DONE]`; TTFT/ITL smoke test shows sub-millisecond flush behavior.
6. **Robustness:** admission 503, slow-consumer abort, disconnect abort,
   graceful shutdown, keep-alive.
7. **Bench:** capture-replay client, then the two-engine `vllm bench serve`
   comparison with the pinned table; results archived with the existing
   provenance discipline.

## 11. Risks / open questions

- **Boost vendoring** is the only new third-party weight; keep it a strict
  header-only subset and record the extraction script in-repo.
- **Detokenizer correctness**: UTF-8 boundaries spanning token deltas;
  phase-1 full re-decode sidesteps it, phase-2 windowed decode needs the
  standard readahead trick and tests with multibyte content.
- **Empty-text chunks** are timestamps by the bench — emit them (never
  suppress special tokens), or TTFT shifts to the wrong token.
- **WSL networking**: localhost only; port 8000 default; WSL's NAT quirks
  documented in the run recipes.
- **Slow-client policy** (abort at 4 MiB pending) needs validation against
  the bench's timeout behavior before it fires on any legitimate run.
- **Open**: do we want `--seed` echoed per request (needs seeded sampling)
  before the first bench publication, or is greedy-only acceptable for the
  first round? Greedy-only recommended.
