# Production HTTP Server Architecture for LLM Inference

## 1. Overall architecture

The HTTP Server is a stateless external API gateway and request-orchestration
layer. It translates external protocols into internal inference commands but
does not execute model inference.

The central design principle is to separate:

- external protocol semantics;
- request lifecycle and tenant policy;
- scheduling decisions;
- model execution; and
- GPU and KV-cache resource management.

```text
                         Control plane
                         ─────────────
                    ┌────────────────────┐
                    │ Model Registry     │
                    │ Auth / RBAC        │
                    │ Quota Service      │
                    │ Configuration      │
                    └─────────┬──────────┘
                              │
                              ▼
┌──────────┐  HTTPS   ┌──────────────────────────────┐
│ Clients  │─────────▶│ HTTP Server / API Gateway    │
└──────────┘          │                              │
                      │ Routing and protocol         │
                      │ Authentication and quotas    │
                      │ Validation and admission     │
                      │ Request Manager              │
                      │ SSE streaming                │
                      └──────────────┬───────────────┘
                                     │ gRPC
                                     ▼
                      ┌──────────────────────────────┐
                      │ Scheduler                    │
                      │                              │
                      │ Priority queues              │
                      │ Continuous batching          │
                      │ Replica selection            │
                      │ Preemption and fairness      │
                      └──────────────┬───────────────┘
                                     │
                    ┌────────────────┴────────────────┐
                    ▼                                 ▼
          ┌──────────────────┐              ┌──────────────────┐
          │ Model Engine A   │              │ Model Engine B   │
          │ model/version X  │              │ model/version Y  │
          └─────────┬────────┘              └─────────┬────────┘
                    │                                 │
          ┌─────────┴────────┐              ┌─────────┴────────┐
          ▼                  ▼              ▼                  ▼
     ┌────────┐         ┌────────┐     ┌────────┐         ┌────────┐
     │ GPU W0 │         │ GPU W1 │     │ GPU W2 │         │ GPU W3 │
     └───┬────┘         └───┬────┘     └───┬────┘         └───┬────┘
         └─────────┬────────┘              └─────────┬────────┘
                   ▼                                 ▼
          ┌─────────────────┐              ┌─────────────────┐
          │ KV Cache Manager│              │ KV Cache Manager│
          └─────────────────┘              └─────────────────┘
```

### Responsibility boundaries

| Component | Responsibilities | Must not own |
|---|---|---|
| HTTP Server | HTTP/SSE, authentication, validation, quotas, admission, request context, and error translation | GPU execution or batch construction |
| Request Manager | External request lifecycle, deadlines, cancellation, streaming buffers, and usage accounting | Model kernels |
| Scheduler | Queue ordering, fairness, batching, replica selection, and preemption | HTTP or OpenAI JSON |
| Model Engine | Model forward passes, sampling coordination, and execution contracts | Authentication or tenant quotas |
| GPU Workers | Execute prefill/decode batches and return token events and execution metrics | Client connections |
| KV Cache Manager | Allocate, reuse, evict, and account KV blocks | API policy |
| Model Registry | Model identity, versions, capabilities, replica state, and routing metadata | Request execution |
| Quota Service | Tenant limits and distributed usage accounting | Construction of individual GPU batches |

The Scheduler owns execution state. The Request Manager owns externally visible
request state. Their states are correlated but deliberately separate.

### HTTP transport decision

The current synchronous `cpp-httplib` implementation is replaced by an
event-driven transport. Two alternatives were evaluated:

1. use Folly coroutines directly over Folly sockets; or
2. use Boost.Beast for HTTP framing and sockets, with the request lifecycle
   expressed as Folly coroutines.

Folly supplies executors, cancellation, asynchronous sockets, futures, and
coroutine primitives. It is not by itself a complete HTTP server. A direct
Folly implementation would therefore have to implement and maintain HTTP/1.1
parsing, persistent connections, pipelining rules, chunked transfer encoding,
body limits, timeout behavior, and SSE framing. Using Proxygen would avoid that
work, but Proxygen is a separate HTTP stack with a substantially larger build
and operational footprint; it is not equivalent to adding Folly coroutines.

Boost.Beast already supplies the required HTTP protocol state machines on top
of Boost.Asio. The selected design is therefore:

```text
Boost.Asio socket and timer executor
              │
              ▼
Boost.Beast HTTP connection/session
              │ narrow awaitable adapter
              ▼
folly::coro::Task request pipeline
              │
              ├── authentication and admission
              ├── scheduler submission
              ├── generation-event consumption
              └── cancellation and cleanup
```

Beast owns bytes, HTTP framing, connection persistence, and transport errors.
Folly coroutines own the application request lifecycle. The adapter between
them is deliberately small and is the only module allowed to translate an Asio
completion into a Folly coroutine completion.

This is preferable to implementing HTTP directly on Folly sockets because it
has comparable event-driven performance, far less protocol code to maintain,
and a narrower correctness and security surface. Its main cost is bridging two
asynchronous ecosystems. That cost is controlled by enforcing executor
affinity and keeping all socket operations on the Asio I/O thread.

## 2. API design

Use `/v1` as the compatibility API. Administrative APIs use a separate prefix
and preferably a separate listener, such as `/admin/v1`.

### Chat completions

`POST /v1/chat/completions`

Representative request:

```json
{
  "model": "qwen2.5-72b-instruct",
  "messages": [
    {"role": "system", "content": "You are helpful."},
    {"role": "user", "content": "Explain continuous batching."}
  ],
  "max_tokens": 256,
  "temperature": 0.7,
  "top_p": 0.95,
  "top_k": 40,
  "repetition_penalty": 1.05,
  "stop": ["</answer>"],
  "stream": true,
  "stream_options": {"include_usage": true},
  "seed": 42,
  "user": "external-user-id"
}
```

Non-streaming response:

```json
{
  "id": "chatcmpl_01J...",
  "object": "chat.completion",
  "created": 1786090000,
  "model": "qwen2.5-72b-instruct",
  "model_version": "2026-07-15",
  "choices": [{
    "index": 0,
    "message": {
      "role": "assistant",
      "content": "Continuous batching..."
    },
    "finish_reason": "stop"
  }],
  "usage": {
    "prompt_tokens": 24,
    "completion_tokens": 91,
    "total_tokens": 115
  }
}
```

### Text completions

`POST /v1/completions`

```json
{
  "model": "base-model",
  "prompt": "The operating system kernel",
  "max_tokens": 128,
  "temperature": 0,
  "stream": false
}
```

The response follows OpenAI's text-completion shape with `choices[].text`.
Batch prompt support must either be fully implemented or rejected explicitly;
the server must never silently accept only the first prompt.

### Embeddings

`POST /v1/embeddings`

```json
{
  "model": "embedding-model-v3",
  "input": ["First document", "Second document"],
  "encoding_format": "float",
  "dimensions": 1024
}
```

Response:

```json
{
  "object": "list",
  "model": "embedding-model-v3",
  "data": [{
    "object": "embedding",
    "index": 0,
    "embedding": [0.012, -0.034]
  }],
  "usage": {"prompt_tokens": 7, "total_tokens": 7}
}
```

Embeddings use a separate scheduler workload class because their batching and
latency characteristics differ from autoregressive generation.

### Models

`GET /v1/models`

Return only models visible to the authenticated tenant:

```json
{
  "object": "list",
  "data": [{
    "id": "qwen2.5-72b-instruct",
    "object": "model",
    "created": 1786090000,
    "owned_by": "platform",
    "status": "ready"
  }]
}
```

### SSE streaming

Response headers:

```http
Content-Type: text/event-stream
Cache-Control: no-cache
Connection: keep-alive
X-Accel-Buffering: no
```

Frames:

```text
data: {"id":"chatcmpl_...","choices":[{"delta":{"role":"assistant"}}]}

data: {"id":"chatcmpl_...","choices":[{"delta":{"content":"Continuous"}}]}

data: {"id":"chatcmpl_...","choices":[{"delta":{},"finish_reason":"stop"}]}

data: {"id":"chatcmpl_...","choices":[],"usage":{"prompt_tokens":24,"completion_tokens":91,"total_tokens":115}}

data: [DONE]
```

Send periodic SSE comments while a request is queued for a long time:

```text
: keepalive
```

### Errors

Use a stable OpenAI-compatible envelope:

```json
{
  "error": {
    "message": "max_tokens exceeds the model limit",
    "type": "invalid_request_error",
    "param": "max_tokens",
    "code": "context_length_exceeded",
    "request_id": "req_01J..."
  }
}
```

| HTTP status | Meaning |
|---:|---|
| 400 | Malformed or invalid request |
| 401 | Missing or invalid credentials |
| 403 | Tenant lacks permission |
| 404 | Unknown or inaccessible model |
| 408 | Request deadline expired before admission |
| 409 | Model lifecycle conflict |
| 413 | Request body too large |
| 422 | Semantically unsupported parameter combination |
| 429 | Rate, quota, concurrency, or capacity limit |
| 499 | Client disconnected; internal/logging only |
| 500 | Internal failure |
| 503 | Model or scheduler unavailable |
| 504 | Inference deadline exceeded |

Include `Retry-After` for retryable `429` and `503` responses.

### Versioning

- `/v1` is the stable external compatibility contract.
- Additive response fields are allowed; breaking changes require `/v2`.
- Internal gRPC APIs are independently versioned, for example
  `inference.scheduler.v1`.
- Compatibility behavior can be selected through headers instead of changing
  existing response semantics.

## 3. Request lifecycle management

The complete request lifecycle is a structured coroutine. A connection session
may serve multiple sequential HTTP/1.1 requests, but each request owns its own
context, deadline, cancellation source, and terminal cleanup guard.

```cpp
folly::coro::Task<void> HttpSession::Run() {
  while (!connection_cancel_.isCancellationRequested()) {
    auto request = co_await transport_.ReadRequest(connection_cancel_);
    if (!request) co_return;

    // HTTP/1.1 pipelining is intentionally serialized per connection. Parallel
    // inference still occurs across connections and scheduler requests.
    const bool keep_alive = request->keep_alive();
    co_await HandleRequest(std::move(*request));

    if (!keep_alive) co_return;
  }
}

folly::coro::Task<void> HttpSession::HandleRequest(HttpRequest request) {
  RequestContext context = request_manager_.NewContext(request);
  auto cleanup = folly::makeGuard([&] {
    request_manager_.Finalize(context.request_id);
  });

  try {
    co_await ValidateRequest(request, context);
    context.principal = co_await auth_.Authenticate(request, context.cancel);
    auto command = co_await request_factory_.Prepare(request, context);
    co_await admission_.Admit(command, context);

    RequestHandle handle = co_await scheduler_.Submit(command, context);
    if (command.stream) {
      co_await StreamResponse(handle, context);
    } else {
      co_await CollectResponse(handle, context);
    }
  } catch (const RequestError& error) {
    co_await WriteError(error, context);
  } catch (const folly::OperationCancelled&) {
    co_await CancelAndClose(context, CancellationReason::kCancelled);
  } catch (...) {
    co_await CancelAndWriteInternalError(context);
  }
}
```

Coroutine ownership follows structured-concurrency rules:

- a request cannot outlive its session unless ownership is explicitly moved to
  the Request Manager;
- spawned work is joined or cancelled before the context is destroyed;
- cancellation is propagated through `folly::CancellationToken`;
- cleanup is idempotent and runs for success, exceptions, timeout, and
  disconnect; and
- no coroutine retains a reference to Beast request or response storage after
  the owning operation completes.

### Request context

```cpp
struct RequestContext {
  RequestId request_id;
  TraceId trace_id;

  TenantId tenant_id;
  PrincipalId principal_id;
  ApiKeyId api_key_id;
  RoleSet roles;

  ModelId requested_model;
  ModelVersion resolved_version;
  WorkloadClass workload;

  TimePoint received_at;
  TimePoint admitted_at;
  TimePoint queued_at;
  TimePoint first_token_at;
  TimePoint completed_at;
  TimePoint deadline;

  Priority priority;
  SamplingParams sampling;

  uint32_t prompt_tokens;
  uint32_t reserved_output_tokens;
  uint32_t generated_tokens;

  RequestState state;
  CancellationSource cancellation;
  std::shared_ptr<EventBuffer> output;
  TraceContext trace;
};
```

Sensitive prompt content should not remain in the context after submission
unless retention is explicitly required.

### State machine

```text
RECEIVED
   │
   ▼
VALIDATING ───────────────▶ REJECTED
   │
   ▼
AUTHENTICATED ────────────▶ REJECTED
   │
   ▼
ADMISSION_PENDING ────────▶ REJECTED / TIMED_OUT
   │
   ▼
QUEUED ───────────────────▶ CANCELLED / TIMED_OUT
   │
   ▼
PREFILLING ───────────────▶ FAILED / CANCELLED
   │
   ▼
DECODING ◀────────────────┐
   │                      │
   └──── next token ──────┘
   │
   ▼
FINISHING
   │
   ├────────▶ COMPLETED
   ├────────▶ CANCELLED
   └────────▶ FAILED
```

Every terminal transition must be idempotent.

### Request IDs

- Generate a globally unique, sortable ID such as UUIDv7 at ingress.
- Never accept a client-supplied ID as authoritative.
- Accept `X-Request-ID` only as a separate correlation value.
- Propagate the platform ID through gRPC metadata, logs, metrics exemplars, and
  traces.
- Scheduler and worker operations use the same ID plus an attempt number.

### Cancellation

Cancellation can originate from a client disconnect, explicit administration,
a deadline, tenant shutdown, model unload, or an internal failure.

```text
HTTP disconnect
  → RequestManager.Cancel(request_id)
  → Scheduler.Cancel(request_id)
  → Model Engine removes the sequence at a safe batch boundary
  → KV Cache Manager releases request blocks
```

Cancellation is idempotent and acknowledged asynchronously. An HTTP thread
must not wait indefinitely for GPU cleanup.

### Timeouts

Maintain separate limits for:

- header and body reads;
- time spent waiting in the queue;
- time to first token;
- inter-token idle time;
- total request time; and
- response writes.

Pass an absolute deadline between services so each hop cannot accidentally
reset the timeout budget.

Timers race the relevant operation through a common helper rather than spawning
detached timeout coroutines:

```cpp
template <typename T>
folly::coro::Task<T> WithDeadline(
    folly::coro::Task<T> operation,
    TimePoint deadline,
    folly::CancellationSource& cancellation);
```

When the deadline wins, it requests cancellation, waits for the operation to
observe it, and maps the result to the appropriate queue, inference, or gateway
timeout. This prevents a timed-out socket or scheduler operation from
continuing against destroyed request state.

## 4. Streaming architecture

```text
GPU worker
   │ TokenBatchEvent
   ▼
Model Engine event aggregator
   │ gRPC stream
   ▼
Scheduler event router
   │ GenerationEvent
   ▼
Request Manager
   │ bounded per-request buffer
   ▼
SSE writer
   │
   ▼
Client
```

Internal generation events should be typed and sequenced:

```protobuf
message GenerationEvent {
  string request_id = 1;
  uint64 sequence_number = 2;
  bytes text_delta = 3;
  repeated int32 token_ids = 4;
  uint32 generated_tokens = 5;
  FinishReason finish_reason = 6;
  Error error = 7;
}
```

### Backpressure and slow clients

A slow HTTP connection must never block GPU execution, scheduler event
processing, or other requests. Use a bounded per-request buffer:

```cpp
struct StreamBufferLimits {
  size_t max_events = 256;
  size_t max_bytes = 1 << 20;
  Duration slow_consumer_timeout = 30s;
};
```

Coalesce adjacent text deltas before declaring the consumer slow. If the buffer
remains full:

1. mark the request as a slow consumer;
2. pause delivery for that request if supported;
3. cancel it after the configured grace period;
4. record `client_too_slow`; and
5. release scheduler and KV-cache resources.

Generated text must never be silently discarded.

The transport adapter detects failed writes and connection shutdown and
signals cancellation immediately. A cleanup callback provides a second path
when the stream writer is blocked waiting for an event. Sequence numbers detect
duplicated or missing internal events.

### Coroutine streaming loop

The SSE writer consumes one bounded asynchronous event stream. It does not run
on a GPU or scheduler thread:

```cpp
folly::coro::Task<void> HttpSession::StreamResponse(
    RequestHandle& handle, RequestContext& context) {
  co_await transport_.WriteSseHeaders(context.cancel);

  while (auto event = co_await handle.events.Next(context.cancel)) {
    if (event->is_delta()) {
      co_await transport_.WriteSseFrame(
          response_encoder_.Delta(*event), context.cancel);
    } else if (event->is_terminal()) {
      co_await transport_.WriteSseFrame(
          response_encoder_.Terminal(*event), context.cancel);
      if (context.include_usage) {
        co_await transport_.WriteSseFrame(
            response_encoder_.Usage(handle.usage()), context.cancel);
      }
      co_await transport_.WriteSseFrame("[DONE]", context.cancel);
      co_return;
    }
  }

  throw UpstreamClosedError();
}
```

Every `WriteSseFrame` suspends until Beast has accepted the bytes for an
asynchronous write. Only one write may be outstanding on a Beast stream. The
bounded `EventBuffer` decouples this socket backpressure from scheduler event
delivery. If the write deadline or slow-consumer deadline expires, the request
is cancelled and its KV allocation is released asynchronously.

### Beast/Folly adapter rules

The bridge must obey the following constraints:

1. Beast stream objects and their buffers remain on their owning Asio executor.
2. The adapter posts initiation to that executor when called elsewhere.
3. Completion resumes exactly one waiting Folly coroutine.
4. Cancellation closes or cancels the underlying socket operation on its Asio
   executor.
5. Adapter state is reference-counted until both completion and cancellation
   paths have quiesced.
6. Application code may inspect Beast HTTP messages but never initiates Beast
   parser, serializer, socket, or timer operations directly.
7. No `blockingWait`, synchronous read, or synchronous write is permitted on an
   I/O thread.

The initial implementation should use one `io_context` per I/O shard with a
small fixed number of threads. Connections are pinned to a shard for their
lifetime. CPU-heavy authentication, JSON processing, and tokenization are
explicitly transferred to bounded CPU executors and then returned to the
connection's I/O executor for writes.

## 5. Scheduler integration

Use bidirectional gRPC streaming for request events and unary RPCs for control
operations:

```protobuf
service InferenceScheduler {
  rpc Submit(SubmitRequest) returns (SubmitResponse);
  rpc Events(EventSubscription) returns (stream GenerationEvent);
  rpc Cancel(CancelRequest) returns (CancelResponse);
  rpc GetStatus(GetStatusRequest) returns (RequestStatus);
  rpc UpdatePriority(UpdatePriorityRequest)
      returns (UpdatePriorityResponse);
}
```

`SubmitRequest` contains:

- request ID and attempt;
- tenant and workload class;
- immutable model version;
- tokenized prompt or tokenizer contract version;
- sampling parameters and stop criteria;
- bounded priority class;
- absolute deadline;
- estimated and reserved tokens; and
- trace context.

Tokenization can run in the Request Manager or a dedicated tokenizer service,
but its revision must match the selected model version.

### Scheduling behavior

- **Continuous batching:** add and remove sequences at decode iteration
  boundaries.
- **Dynamic batching:** group prefill and embedding work by compatibility and
  token budget.
- **Priority scheduling:** use bounded classes, aging, and controlled
  preemption instead of unrestricted numeric priority.
- **Multi-tenancy:** use hierarchical scheduling:

```text
workload class
  → tenant weighted-fair queue
    → priority class
      → request deadline/order
```

No tenant may occupy the complete batch or KV cache indefinitely.

## 6. Multi-model management

The model control plane consists of:

- a Model Registry for desired and observed state;
- a Placement Controller for nodes and GPUs;
- an asynchronous Model Loader;
- a Health Monitor for warm-up and readiness; and
- a Routing Table mapping aliases and versions to ready replicas.

APIs:

```text
GET  /v1/models
GET  /admin/v1/models
GET  /admin/v1/models/{model}/versions/{version}
POST /admin/v1/models/load
POST /admin/v1/models/unload
```

Load request:

```json
{
  "model": "qwen2.5-72b-instruct",
  "version": "2026-07-15",
  "artifact_uri": "s3://models/qwen/2026-07-15/",
  "replicas": 2,
  "tensor_parallel_size": 4,
  "dtype": "bf16"
}
```

Lifecycle operations return an operation ID and run asynchronously:

```json
{"operation_id": "op_01J...", "status": "pending"}
```

Model states:

```text
DISCOVERED → DOWNLOADING → LOADING → WARMING → READY
                                  └──────────▶ FAILED

READY → DRAINING → UNLOADING → UNLOADED
```

Unload removes replicas from routing before draining or cancelling active
requests. Resolve model aliases to immutable versions at admission; a request
must never change versions while running.

## 7. Authentication and multi-tenancy

Authenticate API keys from the Bearer token:

```http
Authorization: Bearer ix_live_...
```

Store only key hashes. Authentication produces:

```cpp
struct Principal {
  TenantId tenant;
  PrincipalId subject;
  ApiKeyId key_id;
  RoleSet roles;
  ScopeSet scopes;
};
```

Use short-lived JWT/OIDC credentials for administrators and service-to-service
calls.

Example roles and scopes:

- `inference.invoke`;
- `models.read`;
- `models.manage`;
- `system.observe`;
- `system.manage`;
- `requests.cancel:own`; and
- `requests.cancel:any`.

Enforce tenant isolation for model visibility, queue capacity, concurrency,
token rate, KV-cache share, scheduling weight, logs, metrics, and admin access.
Do not expose raw tenant IDs as unbounded Prometheus labels.

Use hierarchical token buckets for requests per minute, input and output tokens
per minute, queued requests, and running requests. Reserve the maximum expected
cost at admission and reconcile against actual usage at completion.

## 8. Admission control

Admission runs before expensive queueing or model execution:

```text
body size
→ authentication
→ schema validation
→ model resolution
→ tokenization and counting
→ context-window check
→ tenant quota
→ global capacity
→ scheduler admission
```

Limits include:

- global and per-model queue length;
- global and per-tenant concurrent requests;
- maximum prompt and requested output tokens;
- maximum total context length;
- estimated KV-cache reservation;
- scheduler health; and
- model readiness.

Overload response:

```http
HTTP/1.1 429 Too Many Requests
Retry-After: 2
```

```json
{
  "error": {
    "type": "rate_limit_error",
    "code": "queue_capacity_exceeded",
    "message": "The model queue is full.",
    "request_id": "req_01J..."
  }
}
```

Use `503 model_unavailable` when no healthy model replica exists rather than a
tenant or capacity quota being exceeded.

## 9. Request-parameter validation

| Parameter | Validation |
|---|---|
| `max_tokens` | Integer in `1..model.max_output_tokens`; combined context must fit |
| `temperature` | Finite float, normally `0..2`; zero selects greedy decoding |
| `top_p` | Finite float in `(0,1]` |
| `top_k` | Integer in `0..vocab_size`; zero disables |
| `repetition_penalty` | Finite float, normally `(0,2]`; one disables |
| `stop` | String or bounded array; reject empty values and excessive byte length |
| `seed` | Integer within the internal RNG range |
| `stream` | Boolean |
| `stream_options` | Object; `include_usage` is valid only with streaming |
| `n` | Positive bounded integer; reject until multiple sequences are supported |
| `model` | Required, visible to the tenant, and resolvable to a ready version |

Reject NaN, infinity, overflow, and contradictory parameter combinations. Keep
model-specific capabilities in registry metadata instead of scattering
validation constants through HTTP handlers.

## 10. Token management

Provide `POST /v1/tokenize`:

```json
{
  "model": "qwen2.5-72b-instruct",
  "text": "Hello world",
  "add_special_tokens": false
}
```

```json
{
  "model": "qwen2.5-72b-instruct",
  "model_version": "2026-07-15",
  "token_ids": [9707, 1879],
  "token_count": 2
}
```

An optional `/v1/detokenize` endpoint may be provided with strict limits.

The Token Manager owns:

- tokenizer instances and model-version matching;
- model-specific chat-template application;
- token counting;
- context-window validation;
- explicit prompt-truncation policy;
- admission cost estimation; and
- actual usage reconciliation.

For chat requests:

```text
messages
→ model-version-specific chat template
→ tokenizer
→ prompt token count
→ context validation
```

A character-count heuristic must never be the final admission decision.

## 11. Health and operational APIs

Probe endpoints:

- `GET /health/live`: process event loop and critical threads are alive.
- `GET /health/ready`: schedulers are reachable and required models are ready.
- `GET /health/startup`: initial configuration and dependencies are loaded.

Liveness must not fail merely because a model is temporarily unavailable;
otherwise Kubernetes may create a restart loop.

Administrative APIs:

```text
GET  /admin/v1/status
GET  /admin/v1/gpus
GET  /admin/v1/schedulers
GET  /admin/v1/models
GET  /admin/v1/requests/{id}
POST /admin/v1/requests/{id}/cancel
POST /admin/v1/config/reload
POST /admin/v1/shutdown
```

Graceful shutdown performs these steps:

1. report not-ready;
2. stop accepting new inference requests;
3. drain active requests until a deadline;
4. cancel remaining work;
5. close scheduler streams;
6. flush logs and traces; and
7. exit.

## 12. Observability

### Metrics

Histograms:

- end-to-end, queue, and prefill latency;
- time to first token;
- inter-token latency; and
- cancellation cleanup latency.

Counters:

- requests by endpoint, model, status, and finish reason;
- prompt and completion tokens;
- rejections by reason;
- client disconnects;
- worker and scheduler failures; and
- model load outcomes.

Gauges:

- queued and running requests;
- active streams;
- batch size and batch tokens;
- KV-cache utilization and evictions;
- loaded replicas; and
- GPU memory and utilization.

Derived measurements:

```text
TTFT = first_token_at - received_at
queue_time = admitted_at - queued_at
generation_tps = output_tokens / decode_duration
goodput = successful_tokens / wall_clock_time
```

Use bounded model labels. Never label metrics by request, user, or API key.

### Logging

Emit structured request logs:

```json
{
  "timestamp": "...",
  "severity": "INFO",
  "request_id": "req_01J...",
  "trace_id": "...",
  "tenant": "tenant_hash",
  "model": "qwen2.5-72b-instruct",
  "model_version": "2026-07-15",
  "state": "completed",
  "prompt_tokens": 24,
  "completion_tokens": 91,
  "ttft_ms": 83,
  "latency_ms": 1840
}
```

Keep request logs, internal error logs, and immutable administrative audit logs
separate. Prompt logging is disabled by default.

### Tracing

Propagate W3C `traceparent` and OpenTelemetry baggage across:

```text
HTTP ingress
  → authentication
  → tokenization
  → admission
  → scheduler queue
  → prefill
  → decode
  → stream writes
```

GPU batch spans should link to request spans. A batch contains multiple
requests and therefore cannot be represented accurately as a simple
parent-child trace tree.

## 13. Fault tolerance

| Failure | Handling |
|---|---|
| GPU worker failure | Mark it unhealthy, stop routing, and fail or retry requests that produced no output |
| Scheduler failure | Reconnect the event stream and reject new work with 503 until routing recovers |
| Model loading failure | Keep the old version serving and mark the new operation failed |
| Gateway-to-scheduler timeout | Cancel uncertain submission by request ID; submission must be idempotent |
| Client disconnect | Cancel the request and reclaim KV cache asynchronously |
| Partial generation failure | Emit a terminal error if possible and close; never retry after exposing output |
| KV-cache exhaustion | Preempt according to policy or reject with overload |
| Gateway restart | Active SSE connections end; scheduler cancels orphaned request leases |

Each submitted request has a renewable gateway lease. If the gateway
disappears and the lease expires, the scheduler cancels the orphaned request.
Retries are safe only before output is visible and with an idempotent
`(request_id, attempt)` submission key.

## 14. Kubernetes deployment

```text
External Load Balancer
        │
        ▼
HTTP Gateway Deployment (N replicas)
        │
        ▼
Scheduler Service / Shards
        │
        ▼
Model Engine StatefulSets or custom workloads
        │
        ▼
GPU Nodes
```

### HTTP Server

- Run as a horizontally scalable Deployment.
- Keep request state in memory only for the lifetime of its connection.
- Use a ClusterIP service behind an ingress or L4 load balancer.
- SSE remains on its selected connection; no additional session affinity is
  required.
- Use PodDisruptionBudgets and topology spreading.
- Drain on `SIGTERM` through readiness removal and a termination grace period.

### Scheduler and engines

- Shard by model, tenant group, or capacity domain.
- Use service discovery or a registry-backed routing table.
- Keep GPU workers close to their scheduler to reduce token-event latency.
- Use node selectors, GPU resource requests, and topology-aware placement for
  tensor-parallel groups.

### Rolling upgrades

- Keep internal Protobuf changes backward compatible.
- Upgrade gateways first when they support old and new scheduler versions.
- Drain model replicas before replacement.
- Keep at least one ready replica for every production model version.

### Configuration

- ConfigMaps hold non-secret static configuration.
- Secrets or an external secret manager hold credentials.
- A versioned control-plane service distributes dynamic configuration.
- Validate new configuration before atomic activation.

## 15. Code-level design

```text
server/
├── api/
│   ├── openai_types
│   ├── request_parser
│   ├── response_encoder
│   └── error_mapping
├── transport/
│   ├── io_runtime
│   ├── beast_listener
│   ├── beast_session
│   ├── beast_folly_adapter
│   ├── routes
│   └── sse_writer
├── coroutine/
│   ├── deadline
│   ├── cancellation
│   ├── executor_hop
│   └── task_group
├── middleware/
│   ├── request_id
│   ├── authentication
│   ├── authorization
│   ├── rate_limit
│   └── tracing
├── request/
│   ├── request_context
│   ├── request_manager
│   ├── state_machine
│   └── cancellation
├── admission/
│   ├── admission_controller
│   ├── quota
│   └── capacity_policy
├── tokenization/
│   ├── tokenizer_registry
│   ├── chat_template
│   └── token_accounting
├── streaming/
│   ├── event_buffer
│   ├── event_router
│   └── backpressure
├── scheduler_client/
│   ├── scheduler_client
│   ├── event_subscription
│   └── retry_policy
├── model_registry/
│   ├── registry
│   ├── routing_table
│   └── lifecycle_client
├── auth/
│   ├── principal
│   ├── api_key_store
│   └── rbac
├── observability/
│   ├── metrics
│   ├── logging
│   └── tracing
└── admin/
    ├── model_handlers
    ├── system_handlers
    └── shutdown
```

Key interfaces:

```cpp
class RequestManager {
 public:
  StatusOr<RequestHandle> Create(ValidatedRequest request,
                                 Principal principal);
  Status Cancel(RequestId id, CancellationReason reason);
  StatusOr<RequestSnapshot> GetStatus(RequestId id);
};

class AdmissionController {
 public:
  AdmissionDecision Evaluate(const RequestContext&,
                             const CapacitySnapshot&);
  void Reconcile(const RequestUsage&);
};

class SchedulerClient {
 public:
  virtual folly::coro::Task<SubmitResult> Submit(
      ScheduledRequest, folly::CancellationToken) = 0;
  virtual folly::coro::AsyncGenerator<GenerationEvent&&> Events(
      RequestId, folly::CancellationToken) = 0;
  virtual folly::coro::Task<void> Cancel(RequestId) = 0;
  virtual folly::coro::Task<void> UpdatePriority(
      RequestId, PriorityClass) = 0;
};

class TokenizationService {
 public:
  StatusOr<TokenizedPrompt> TokenizeCompletion(ModelVersion, string_view);
  StatusOr<TokenizedPrompt> TokenizeChat(ModelVersion,
                                         span<const ChatMessage>);
};

class EventBuffer {
 public:
  PushResult TryPush(GenerationEvent);
  folly::coro::Task<std::optional<GenerationEvent>> Next(
      folly::CancellationToken);
  void Close(TerminalStatus);
};
```

HTTP handlers remain thin:

```cpp
folly::coro::Task<void> ChatCompletions(HttpRequest request,
                                        HttpResponseWriter& writer,
                                        RequestContext& context) {
  Principal principal =
      co_await auth.Authenticate(request, context.cancel);
  ChatRequest parsed = api.ParseChat(request.body);
  RequestHandle handle =
      co_await request_manager.Create(parsed, principal, context);

  if (parsed.stream) {
    co_await sse_writer.Stream(handle, writer, context.cancel);
  } else {
    co_await response_writer.Collect(handle, writer, context.cancel);
  }
}
```

## 16. Technology recommendations

### Alternatives evaluated

#### Alternative A: Folly coroutines directly

This alternative uses `folly::EventBase`, `AsyncServerSocket`, and
`AsyncTransport` for network I/O, with `folly::coro::Task` for each connection
and request.

Advantages:

- one executor, cancellation, and coroutine ecosystem;
- no bridge between Asio and Folly completion models;
- strong integration with planned Folly queues and other Folly utilities; and
- precise control over allocation, thread affinity, and connection scheduling.

Disadvantages:

- Folly alone does not provide the complete HTTP server behavior required by
  this design;
- implementing HTTP directly creates a large security and correctness surface;
- persistent connections, malformed-message handling, chunked bodies, limits,
  and protocol upgrades become InferX responsibilities;
- conformance and fuzzing costs are much greater than the inference-specific
  value provided; and
- adopting Proxygen to fill this gap introduces another large framework and a
  materially heavier build than either Folly or Beast alone.

This option can offer excellent performance, but HTTP parsing is not the
bottleneck in LLM serving. The additional protocol ownership is not justified.

#### Alternative B: Boost.Beast with Folly coroutines

This alternative uses Boost.Asio and Beast for sockets, timers, HTTP parsing,
serialization, keep-alive, and chunked writes. A narrow adapter turns these
operations into `folly::coro::Task` operations used by the application layer.

Advantages:

- established, incremental HTTP parsing and serialization;
- body limits, buffer ownership, connection persistence, and transport errors
  remain in a dedicated protocol library;
- the application request flow remains linear and cancellation-aware;
- migration can preserve the existing OpenAI parsing and response encoders;
- much less custom protocol code and a smaller test surface; and
- it matches InferX's existing architectural direction toward Beast.

Disadvantages:

- Asio and Folly use different executors and cancellation models;
- the bridge must handle completion-versus-cancellation races correctly;
- careless executor hopping can add latency or resume a coroutine on the wrong
  thread; and
- the build gains both Boost and Folly rather than only one dependency family.

### Comparison

| Criterion | Direct Folly sockets | Beast + Folly coroutines |
|---|---|---|
| HTTP performance | Potentially excellent | Excellent; overhead is negligible relative to inference |
| Streaming scalability | Excellent | Excellent |
| Protocol correctness | Must be implemented or add Proxygen | Supplied by Beast |
| Coroutine integration | Native | Requires a narrow adapter |
| Build complexity | High with Folly; very high with Proxygen | High but bounded |
| Maintenance burden | Very high for custom HTTP | Moderate |
| Security surface | Large, InferX-owned parser/state machine | Smaller, library-owned protocol machinery |
| Migration from `cpp-httplib` | Full replacement | Incremental route-by-route replacement |
| Fit with current InferX design | Partial | Strong |

### Recommendation

Use **Boost.Beast as the HTTP layer with Folly coroutines for request
processing**.

The deciding factor is maintainability rather than raw throughput. Both options
can handle substantially more concurrent connections than the GPU scheduler can
serve. Beast avoids turning InferX into an HTTP protocol implementation while
Folly coroutines provide structured cancellation and readable asynchronous
application flow.

The Beast/Folly bridge must remain private to `server/transport`. If bridge
complexity becomes disproportionate during implementation, the approved
fallback is Beast with native Boost.Asio C++20 awaitables while retaining Folly
for queues only. That fallback preserves the selected HTTP layer and removes an
executor boundary without changing Request Manager or Scheduler contracts.

Before removing `cpp-httplib`, validate the decision with a transport benchmark
that excludes model execution and covers:

- at least 10,000 idle keep-alive connections;
- at least 1,000 concurrent synthetic SSE streams;
- response-write throughput for small token-sized frames;
- p50, p95, and p99 event-to-socket scheduling latency;
- resident memory per idle and active connection;
- cancellation latency under disconnect storms; and
- CPU consumption with and without the Beast/Folly adapter.

Compare the selected implementation with a Beast-native Asio-coroutine
reference. If the Folly adapter adds more than 5% CPU at equal throughput or a
repeatable 10 microseconds to p99 event-to-write scheduling latency, use the
native Asio fallback. These are transport acceptance limits, not inference
latency objectives; end-to-end TTFT remains dominated by admission, prefill,
and scheduling.

### Runtime and dependency choices

| Concern | Selection |
|---|---|
| Language | C++20 |
| HTTP protocol | Boost.Beast |
| Socket and timer runtime | Boost.Asio |
| Application coroutines | `folly::coro::Task` and `AsyncGenerator` |
| Cancellation | `folly::CancellationSource` and `CancellationToken` |
| Cross-thread queues | Folly bounded queues |
| Scheduler RPC | gRPC and Protobuf when process separation is introduced |
| External streaming | HTTP/1.1 SSE |
| Metrics | Prometheus/OpenMetrics |
| Tracing | OpenTelemetry |
| Logs | Structured JSON |

Pinned source revisions:

- Boost `boost-1.91.0` (`1a80576db6b70828803819fb6925132193bc5d0e`);
- Folly `v2026.08.03.00` (`72f7ac6f84243a73f597fab286e3dd049511b344`).

Both are required by the HTTP server. InferX consumes them only through
`inferx::beast` and `inferx::folly_coro`; application targets must not link the
upstream targets directly.

HTTP/2 is not supplied by Beast itself. Initial scope remains HTTP/1.1 with SSE
behind an ingress or load balancer that may terminate HTTP/2. Native end-to-end
HTTP/2 should be added only through a deliberate protocol-layer change, not by
extending the Beast HTTP/1 implementation ad hoc.

### Communication

- Use external HTTP/SSE for broad client compatibility.
- Use gRPC bidirectional streams for scheduler events.
- Use Protobuf for stable, versioned internal contracts.
- Use a control-plane registry for desired model state.
- Batch token events across requests and short time windows; do not issue one
  RPC per token.

## 17. Repository-specific implementation plan

This section is based on the InferX code as it exists on 2026-08-07. It is an
incremental migration plan, not a greenfield layout. Each phase must leave a
buildable server and should land independently.

### 17.1 Current implementation baseline

The existing code already provides useful parts of the target architecture:

- `cmake/InferXDependencies.cmake` pins Boost 1.91.0 and Folly
  2026.08.03.00 and exposes only `inferx::beast` and
  `inferx::folly_coro` to first-party targets.
- `src/server/http_server.cc` uses asynchronous Beast reads, writes, accepts,
  body limits, keep-alive, chunked SSE responses, socket deadlines, a bounded
  active-request count, SHA-256 API-key matching, and disconnect-driven
  generation cancellation.
- `include/inferx/server/transport/response_writer.h` defines the coroutine
  response boundary, and
  `include/inferx/server/transport/beast_folly_adapter.h` contains the first
  completion/cancellation race adapter.
- `src/api/openai.cc` is host-only and already owns request parsing, response
  JSON, SSE framing, stop-string matching, and usage encoding for chat and text
  completions.
- `src/server/engine.cc` supplies an in-process execution adapter: it owns the
  tokenizer, scheduler, model runner, generation loop, metrics, and a blocking
  `Generation::Next` event queue.
- `src/scheduler/scheduler.cc` is host-only and already implements FCFS
  continuous batching, chunked prefill, KV allocation, recompute preemption,
  prefix caching, cancellation, token deltas, and terminal completions.

The migration must explicitly address these current limitations:

1. `HttpServer::Impl::Session` combines transport, routing, authentication,
   validation, tokenization, engine submission, response encoding, metrics,
   deadlines, and cleanup in one callback-driven type.
2. `StartStream` and `WriteChunk` post an asynchronous operation to the I/O
   executor and then block an application-pool thread on `std::future::get`.
   This avoids blocking an I/O thread, but it is not structured asynchronous
   request processing and consumes one application thread per active stream.
3. `Generation::Next` is a condition-variable-backed blocking interface with
   an unbounded deque. A slow client can therefore retain generated output and
   an application thread without a defined event/byte limit.
4. Request state is stored as fields on a reusable connection session. There
   is no independent request state machine, terminal-transition guard, tenant
   identity, absolute deadline, or queryable request snapshot.
5. The HTTP server depends directly on the concrete, GPU-owning `Engine`.
   Protocol and lifecycle tests consequently cannot replace execution with a
   fake through the public server constructor.
6. Authentication returns only a Boolean. There is no principal, tenant,
   scope, RBAC, per-tenant rate limit, token quota, or capacity snapshot.
7. Model routing is a string comparison against one loaded model, and the
   Qwen2 chat template is selected directly by the handler.
8. There is no embeddings endpoint, administrative listener, scheduler RPC,
   distributed model registry, or tracing implementation.
9. `tests/unit` is empty and the top-level build currently defines no first-
   party test targets. Test infrastructure is therefore the first required
   implementation dependency, not a final cleanup task.

The current Beast server remains the wire-compatibility reference until the
new coroutine path passes the compatibility and load gates below. Do not
replace working behavior and architecture simultaneously without a test at
the boundary being changed.

### 17.2 Target dependency graph and build boundaries

Use the directory layout in section 15 with public contracts under
`include/inferx/server/<module>/` and implementations under
`src/server/<module>/`. The intended dependency direction is:

```text
server/api                    server/auth      server/model_registry
    │                              │                    │
    └──────────────┬───────────────┴────────────┬───────┘
                   ▼                            ▼
             server/request              server/admission
                   │                            │
                   └──────────────┬─────────────┘
                                  ▼
                         request service/handlers
                           │                 │
                           ▼                 ▼
                 server/tokenization  server/scheduler_client
                           │                 │
                           └────────┬────────┘
                                    ▼
                         server/streaming events
                                    │
                                    ▼
                         server/transport writer
```

`server/transport` may depend on Beast, Asio, Folly coroutine primitives, and
generic request handlers. It must not depend on `Engine`, CUDA, a model class,
the scheduler implementation, OpenAI domain structs, authentication policy,
or tokenization. Conversely, no module outside `server/transport` may initiate
a Beast parser, serializer, socket, or timer operation.

Split CMake targets so host-only code is built even when
`INFERX_CUDA_MEETS_FLOOR` is false:

- `inferx::http_transport`: Beast/Folly adapter, I/O runtime, listener,
  session, response writer, routing primitives, and SSE byte writer;
- `inferx::server_protocol`: server API parsing/encoding and error mapping,
  linking the existing `inferx::api` during migration;
- `inferx::request_runtime`: request context/state machine, admission,
  streaming buffer, auth contracts, registry contracts, and handler pipeline;
- `inferx::engine_client`: the in-process adapter from the scheduler-client and
  tokenizer-service contracts to the existing `Engine`; and
- `inferx::server`: the CUDA-gated composition root and executable support.

Upstream Boost and Folly targets must remain private behind the two existing
InferX aliases. CUDA libraries must appear only below `inferx::engine_client`
or the final composition target. Add a configure-time or link-time test that a
host-only build can compile protocol, lifecycle, admission, streaming, and
transport unit tests without `inferx::model` or `inferx::kernels`.

### 17.3 Phase 0: establish tests and capture current wire behavior

Before moving production code, add `tests/CMakeLists.txt`, enable it through
`include(CTest)`/`BUILD_TESTING`, and provide small host-only test executables
using the repository's pinned GoogleTest. Start with:

- `tests/unit/api/openai_test.cc`: valid and invalid chat/completion parsing,
  field type failures, sampling bounds, JSON escaping, usage chunks, multiline
  SSE framing, UTF-8 stop-string boundaries, and exact response fixtures;
- `tests/unit/scheduler/scheduler_test.cc`: admission, mixed prefill/decode,
  cancellation in waiting and running states, completion reasons, preemption,
  prefix-cache accounting, context limits, and KV no-leak invariants;
- `tests/unit/transport/beast_folly_adapter_test.cc`: synchronous completion,
  asynchronous completion, initiation exception, Asio error, cancellation
  before initiation, completion/cancellation races, and late callback lifetime;
  and
- `tests/integration/http_compat_test.cc`: start the existing server on port
  zero with a lightweight fake execution seam introduced solely for the test,
  then capture status, headers, JSON, keep-alive, body-limit, method handling,
  authentication, SSE chunks, `[DONE]`, usage, timeout, and disconnect behavior.

If a fake cannot be introduced without prematurely redesigning `HttpServer`,
first extract the minimal `GenerationBackend` interface described in phase 3
and keep `Engine` as its only production implementation. Do not make HTTP
compatibility tests require a checkpoint or GPU.

Exit criteria:

- `ctest` discovers and runs the new host-only suites;
- existing OpenAI behavior is represented by checked-in semantic fixtures,
  not fragile comparisons of JSON object member order unless order is part of
  a deliberate compatibility promise; and
- ASan/UBSan can run the host-only tests with `INFERX_ENABLE_ASAN=ON`.

### 17.4 Phase 1: finish and verify the asynchronous transport core

Implement the private transport files:

```text
src/server/transport/
  io_runtime.cc
  beast_listener.cc
  beast_session.cc
  beast_folly_adapter.cc
  routes.cc
  sse_writer.cc
```

Define corresponding public or private headers as appropriate. Keep adapter
details private; only `ResponseWriter`, `RequestHandler`, server configuration,
and lifecycle controls should be public contracts.

Required changes:

1. Extend `AwaitAsio` to support operations with no result and ensure the
   cancellation callback cannot post after the owning Asio executor has begun
   shutdown. Preserve the single atomic winner and shared late-callback state.
2. Implement a concrete Beast `ResponseWriter`. It owns serializer and chunk
   state on the connection strand, rejects overlapping writes, and maps closed
   sockets and timeout errors into stable `Status` values.
3. Implement `IoRuntime` as explicit I/O shards. A connection is assigned once
   and remains on one strand/executor for its lifetime.
4. Move accept, incremental read, body limit, keep-alive loop, shutdown, and
   session registry out of `http_server.cc` without changing route behavior.
5. Express each connection as a coroutine that serializes HTTP/1.1 requests.
   Do not support concurrent pipelined responses in the first implementation.
6. Replace every `std::promise`, `std::future::get`, synchronous Beast call,
   and blocking write path with `co_await` through the adapter.
7. Make header-read, body-read, response-write, and idle keep-alive timeouts
   separate configuration values and cancellation sources.

Exit criteria:

- transport tests cover fragmented input, malformed/chunked bodies, body
  limits, keep-alive, peer shutdown, write failure, timer expiry, and server
  shutdown with active sessions;
- ThreadSanitizer or deterministic stress tests execute at least 100,000
  completion-versus-cancellation races without double resume, use-after-free,
  or hung task; and
- source checks find no `blockingWait`, `.get()` on a future, synchronous
  socket read/write, or direct Beast operation outside `server/transport`.

### 17.5 Phase 2: extract routing and the protocol layer

Move route selection out of the session. `routes.cc` should match method and
normalized path, apply route metadata such as body limits and authentication
requirements, and dispatch a `RequestHandler`. It must not parse OpenAI JSON.

Split the current `inferx::api` implementation along the design boundaries
without breaking its existing callers:

- `server/api/openai_types`: request, response, usage, finish-reason, and
  stream-option domain types;
- `server/api/request_parser`: JSON-to-domain conversion and syntactic field
  validation;
- `server/api/response_encoder`: JSON and SSE payload generation;
- `server/api/error_mapping`: `Status`/typed request error to HTTP status,
  OpenAI error type, parameter, code, `Retry-After`, and request ID.

Initially, `include/inferx/api/openai.h` may be a compatibility facade over
these smaller contracts. Preserve current chat/text/tokenize behavior while
adding parser support for all documented fields. Unsupported combinations
must return a typed 422 error; they must never be silently ignored. In
particular, reject completion prompt arrays until batch-prompt execution is
implemented. Add `/v1/embeddings` types and parser now, but leave the route
disabled until phase 9 supplies execution.

Exit criteria:

- protocol tests need neither Beast nor CUDA;
- transport tests use a trivial fake handler and know nothing about OpenAI;
- every documented error path has a stable `param` and `code`; and
- compatibility tests show no unintended regression in existing endpoints.

### 17.6 Phase 3: define execution-neutral request and event contracts

Introduce strong request identifiers and lifecycle types under
`server/request`:

- `RequestId`, generated as a standards-compliant UUIDv7 or equivalently
  sortable 128-bit identifier and formatted with the `req_` prefix;
- distinct `TenantId`, `PrincipalId`, `ApiKeyId`, `ModelId`, and
  `ModelVersion` types to prevent accidental interchange;
- `RequestContext` containing identity, immutable resolved model/version,
  workload class, priority, absolute deadline, timestamps, accounting,
  cancellation source, trace context, and state;
- `RequestState` and a checked transition table; and
- `RequestSnapshot`, `CancellationReason`, and terminal status.

Then introduce the execution seam under `scheduler_client`:

```cpp
class SchedulerClient {
 public:
  virtual folly::coro::Task<StatusOr<SubmitResult>> Submit(
      ScheduledRequest, folly::CancellationToken) = 0;
  virtual folly::coro::AsyncGenerator<GenerationEvent&&> Events(
      RequestId, folly::CancellationToken) = 0;
  virtual folly::coro::Task<Status> Cancel(
      RequestId, CancellationReason) = 0;
  virtual folly::coro::Task<StatusOr<RequestSnapshot>> GetStatus(
      RequestId, folly::CancellationToken) = 0;
  virtual folly::coro::Task<Status> UpdatePriority(
      RequestId, PriorityClass, folly::CancellationToken) = 0;
};
```

`GenerationEvent` must include request ID, monotonically increasing sequence
number, text delta or embedding result, generated token count, terminal reason,
usage, and typed error. Blocking and streaming HTTP responses must consume this
same event type.

Add `InProcessSchedulerClient` as a compatibility adapter around the current
`Engine::Submit` and `Generation` APIs. During this phase it may use a bounded
CPU executor to wait on `Generation::Next`, but it must copy events into the
new bounded event path and propagate cancellation. This isolates the temporary
blocking bridge; handlers and transport must not call `Engine` or `Generation`
directly.

Exit criteria:

- `HttpServer::Create` accepts abstract service dependencies (or a single
  service bundle), while the executable composes the in-process adapter;
- fake scheduler and tokenizer implementations can drive a complete HTTP
  request without CUDA; and
- event sequence gaps, duplicates, terminal-after-terminal, and events after
  cancellation are detected and tested.

### 17.7 Phase 4: implement Request Manager and structured lifecycle

Implement `RequestManager` as the sole owner of externally visible request
state. Use a sharded map or otherwise bounded concurrent registry; completed
snapshots need a configured retention time and maximum count.

For each request:

1. create and register the context at ingress;
2. transition through validation, authentication, admission, queueing, and
   generation using checked methods rather than direct state writes;
3. run one structured coroutine whose children are joined or cancelled before
   context destruction;
4. race operations against the context's absolute deadline with
   `WithDeadline`, never detached timer tasks;
5. make `Cancel` and terminal finalization idempotent;
6. submit scheduler cancellation once, asynchronously, on disconnect,
   deadline, shutdown, or policy cancellation; and
7. erase prompt text and bearer-token material as soon as parsing/tokenization
   no longer needs them.

Use one cleanup guard to reconcile admission/accounting, close the event
buffer, cancel unfinished scheduler work, record terminal metrics, and retain
the final snapshot. Connection shutdown requests cancellation but does not own
the remaining scheduler/KV cleanup lifetime.

Exit criteria:

- table-driven tests cover every legal and illegal state transition;
- cancellation is idempotent from every non-terminal state;
- deadline tests use a controllable clock and cover queue, first-token,
  inter-token, total, and write timeouts; and
- concurrent `Cancel`, terminal event, and disconnect always produce exactly
  one terminal state and one accounting reconciliation.

### 17.8 Phase 5: implement model-aware tokenization

Define a `TokenizationService` contract that receives immutable
`ModelVersion`, not an `Engine` reference. Implement a registry of immutable
tokenizer/template bundles loaded during model registration. Each bundle owns
the tokenizer artifact revision, chat-template identity, special tokens,
context limit, and capabilities.

Adapt the existing tokenizer code rather than duplicating it:

- continue cloning `tokenizer::Tokenizer` when the backend requires
  request-confined state;
- move hard-coded `ApplyQwen2ChatTemplate` selection out of HTTP handlers;
- validate prompt plus reserved output tokens against the selected model
  version before scheduler submission;
- return exact prompt-token accounting with the tokenized command; and
- preserve the distinction between user text (`kAsText`) and trusted template
  control tokens (`kAsControl`).

Cache only immutable loaded artifacts and bounded reusable clones. Do not
cache tenant prompt text or formatted chats by default.

Exit criteria:

- completion and chat tokenization tests run without the model engine;
- two model versions can select different templates/tokenizer revisions;
- special-token injection, empty prompts, malformed UTF-8, context overflow,
  and tokenizer failure have stable API errors; and
- HTTP handlers contain no template or tokenizer calls.

### 17.9 Phase 6: bounded streaming and slow-client handling

Replace the unbounded `Generation::events_` exposure with `EventBuffer` in the
request layer. The initial implementation should use both an event-count and a
byte-count bound, coalesce adjacent text deltas, and expose an asynchronous
`Next` operation supporting Folly cancellation.

`EventRouter` validates request ID and sequence number and performs a
non-blocking `TryPush`. `BackpressureController` records the first-full time,
requests upstream pause if the backend supports it, and cancels the request
after `slow_consumer_timeout`. Until upstream pause exists, the explicit policy
is to cancel rather than discard text.

Implement `SseWriter` on top of `ResponseWriter`:

- send the current headers plus `Connection: keep-alive` where valid;
- emit an initial assistant role for chat;
- consume generation events and encode deltas;
- emit a terminal choice, optional usage chunk, and `[DONE]` exactly once;
- emit periodic `: keepalive` comments while queued or while an allowed
  upstream idle period is in progress; and
- cancel immediately on failed or expired writes.

Non-streaming collection must use the same buffer and enforce a configured
maximum response size rather than appending without bound.

Exit criteria:

- tests exercise full-by-count, full-by-bytes, coalescing, slow-consumer
  timeout, upstream close, duplicate terminal, disconnect during write, and
  cancellation while waiting for an event;
- generated text is never silently dropped; and
- a synthetic stalled client does not consume an application thread or delay
  event delivery to another request.

### 17.10 Phase 7: authentication, authorization, and admission

Extract the current SHA-256 key comparison into `ApiKeyStore` and return a
`Principal`. Keep constant-time comparison and never log raw tokens or full
hashes. Add an atomic snapshot-based key configuration so rotation does not
interrupt active requests.

Add middleware in this order:

```text
request ID → tracing context → authentication → authorization
           → syntactic parsing → model resolution → semantic validation
           → quota/rate checks → capacity admission
```

Implement local, interface-driven versions first:

- RBAC and scopes per route;
- token-bucket request/token rate limits per tenant and API key;
- concurrency reservations released by the Request Manager cleanup guard;
- prompt plus maximum-output token reservations reconciled with actual usage;
  and
- a capacity policy using queue depth, event-buffer memory, scheduler health,
  and model readiness.

The global `max_active_requests` remains a final process safety limit but is no
longer the tenant policy. Decisions must return a stable reason and retry
advice. Define distributed quota interfaces now; add a remote implementation
only when multi-replica deployment requires it.

Exit criteria:

- authentication bypass is limited to an explicit development configuration;
- route scope, tenant isolation, key rotation, rate refill, concurrent
  reservation races, rejection rollback, and usage reconciliation are tested;
- retryable 429/503 responses include bounded `Retry-After`; and
- metrics do not use tenant, API-key, model alias, or request ID as unbounded
  labels.

### 17.11 Phase 8: switch all existing routes to thin coroutine handlers

Implement handlers for health, readiness, models, tokenize, chat completions,
text completions, metrics, and the legacy stats endpoint. The chat/text
handlers should only:

1. authenticate through middleware-provided context;
2. parse and validate a protocol request;
3. resolve a model version;
4. ask the request service to create/submit the request; and
5. select streaming or collection response presentation.

Keep `/stats` as a documented legacy route during migration. Health routes
must not require authentication. Readiness must incorporate listener state,
configuration validity, scheduler connectivity, and at least one ready model;
liveness must not fail merely because a model is unavailable.

Run old and new handler implementations behind an internal configuration flag
for compatibility testing. Do not expose two public ports or route prefixes.
After semantic fixture equivalence and load acceptance, remove the old
`HttpServer::Impl::Session` handler methods and make the coroutine path the
only path.

Exit criteria:

- `src/server/http_server.cc` is reduced to composition/lifecycle code or
  removed;
- no handler includes `engine.h`, scheduler implementation headers, tokenizer
  backend headers, OpenSSL, Asio, or Beast operation headers; and
- all current endpoints retain intentional wire behavior, with documented
  changes called out in release notes.

### 17.12 Phase 9: embeddings and workload-aware scheduling

Add embeddings only after the common lifecycle is stable. Extend the
scheduler-client contract with a typed embedding command/result while keeping
the HTTP handler independent of execution details. The model registry must
advertise embedding capability, input limits, dimensions, and supported
encoding formats.

For the current in-process engine, either implement a genuine embedding model
path and separate scheduler workload class or return a clear model-capability
error. Never simulate embeddings from generation logits and never accept only
the first item of a batch input.

Exit criteria:

- string and string-array inputs preserve indices and token accounting;
- generation and embedding admission/batching policies are isolated; and
- unsupported dimensions, encodings, models, and batch sizes return stable
  errors.

### 17.13 Phase 10: model registry and administrative plane

Implement an in-memory registry first, populated from the model loaded by the
existing executable. Store immutable versions, capabilities, tokenizer
revision, lifecycle state, replicas, and tenant visibility. Model aliases are
resolved to immutable versions before tokenization and admission.

Then add `/admin/v1` on a separately configurable listener. Require a distinct
administrative scope and default the listener to disabled. Load/unload APIs
return operation IDs and update the state machine asynchronously. Unload first
removes replicas from routing, then drains until a deadline, then cancels if
policy permits, and only then releases model resources.

The initial single-engine composition may expose one permanently loaded model
and return `unimplemented` for load/unload, but the registry and handler
contracts must not assume that limitation.

Exit criteria:

- tenant-filtered `/v1/models`, alias/version stability, lifecycle transitions,
  drain behavior, operation lookup, and admin authorization are tested; and
- no request can change resolved model version after admission.

### 17.14 Phase 11: observability and configuration

Move HTTP metrics out of ad hoc rendering in `http_server.cc` into
`server/observability`. Reuse `inferx::observe` as the underlying registry.
Instrument request count, active requests, queue duration, TTFT, inter-token
latency, duration, response bytes, cancellation reasons, rejection reasons,
event-buffer utilization, slow consumers, scheduler RPC state, and model
readiness.

Add structured JSON logs at ingress, admission, first token, terminal state,
and exceptional cancellation. Default logs contain IDs and counts, not prompt,
generated text, authorization headers, or raw client correlation values. Add
OpenTelemetry behind a narrow tracing interface and propagate W3C trace
context and the platform request ID through scheduler metadata.

Replace direct command-line-only configuration assembly with a validated
`ServerConfig` snapshot. Static listener/thread settings are startup-only;
keys, quotas, route policies, registry state, and observability sampling may be
atomically reloaded. Reject an invalid snapshot without partially applying it.

Exit criteria:

- metric names and bounded labels have unit tests;
- log redaction tests use deliberately sensitive fixtures;
- traces connect ingress, tokenization, admission, scheduler, first-token, and
  response spans; and
- invalid reloads preserve the last valid configuration.

### 17.15 Phase 12: process-separated scheduler client

Do this only after the in-process interface has stabilized through production
use. Define `inference.scheduler.v1` Protobuf messages from the existing
domain contracts; do not expose C++ implementation types. Carry the absolute
deadline, request/attempt ID, tenant, immutable model version, workload,
priority, tokenized prompt, sampling parameters, tokenizer contract revision,
and trace context.

Implement gRPC submission/control calls and a streamed event subscription.
Reconnect with bounded exponential backoff and jitter. A request ID plus
attempt number makes submission and cancellation idempotent. On an ambiguous
submit result, query status before retrying. Event sequence numbers detect
loss/replay; never silently splice event streams from different attempts.

Keep `InProcessSchedulerClient` as a development and correctness-reference
backend. Select the backend in the composition root; no handler or Request
Manager code changes between them.

Exit criteria:

- contract compatibility tests run client and server at adjacent supported
  protocol versions;
- scheduler restart, stream interruption, duplicate submit/cancel, deadline
  expiry, and gateway shutdown are tested; and
- horizontal gateway replicas own no GPU, scheduler queue, or KV state.

### 17.16 Final migration and removal criteria

Remove temporary compatibility layers only when all of the following hold:

1. The coroutine server passes protocol fixtures for all existing endpoints.
2. Blocking and streaming generation consume the same bounded event
   abstraction.
3. No socket, I/O-executor thread, application-pool thread, or request context
   is retained after completion/cancellation stress tests.
4. Disconnect-to-scheduler-cancel latency and scheduler-cancel-to-KV-release
   latency are measured separately and meet an agreed operational SLO.
5. Host-only protocol, transport, lifecycle, auth, admission, registry, and
   fake-scheduler tests run in CI; GPU integration tests remain a separate
   suite.
6. The transport benchmark covers 10,000 idle keep-alive connections, 1,000
   concurrent synthetic SSE streams, token-sized write throughput,
   p50/p95/p99 event-to-write latency, memory per connection, disconnect
   storms, and graceful shutdown.
7. The Beast/Folly bridge stays within the section 16 CPU and p99 overhead
   limits versus the Beast-native Asio-coroutine reference. Otherwise use the
   approved native-Asio fallback behind unchanged higher-level contracts.
8. Chat templates are selected by immutable model version, and HTTP handlers
   contain no tokenizer, scheduler, or GPU logic.
9. Cancellation, deadline, terminal transition, accounting reconciliation,
   event-buffer close, and response completion are each idempotent.
10. API keys, prompt text, generated text, and high-cardinality tenant/request
    data do not leak into logs or metric labels.
11. The old monolithic session handler, blocking `Generation::Next` bridge,
    and any obsolete `cpp-httplib` files/dependency references are removed.

The recommended delivery units are phases 0 through 8, in order. Phases 9
through 12 add new product capabilities and deployment topology and should not
delay replacing the blocking per-stream application-thread behavior in the
current server.

## 18. DeepSeek-V2-Lite support: design and implementation plan

This section is based on the InferX code as it exists on 2026-08-08. It plans
the third served architecture, `DeepseekV2ForCausalLM`, which is also the
first checkpoint to exercise the MLA attention layer built in M9. Like
section 17 it is an incremental plan: each phase leaves a buildable tree, and
correctness phases are separated from performance phases deliberately.

### 18.1 Why this model and what it exercises

DeepSeek-V2-Lite is the smallest checkpoint that combines the two M9 layers —
MLA attention and a routed MoE FFN — with a real tokenizer, a real chat
template, and published reference logits. Serving it closes the two open M9
items at once: "wiring MLA into a served checkpoint" and the first real user
of the bf16 stacked-expert MoE path (gpt-oss uses only the MXFP4 path).

Model shape, from its `config.json`:

| Property | Value |
|---|---|
| Architecture | `DeepseekV2ForCausalLM` |
| Parameters | 15.7 B total, ~2.4 B active per token |
| Layers / hidden / heads | 27 / 2048 / 16 (no GQA split; MLA) |
| MLA | `kv_lora_rank` 512, `q_lora_rank` **null** (no Q compression), `qk_nope_head_dim` 128, `qk_rope_head_dim` 64, `v_head_dim` 128 |
| Effective QK head dim | 192 (128 nope + 64 rope) |
| MoE | 64 routed experts, top-6, `scoring_func` softmax, `topk_method` greedy, `norm_topk_prob` false, `routed_scaling_factor` 1.0 |
| Shared experts | 2, **ungated**, stored as one fused MLP of width 2 × 1408 = 2816 |
| Dense layers | `first_k_dense_replace` 1: layer 0 is a dense FFN at `intermediate_size` 10944; layers 1–26 are MoE (`moe_layer_freq` 1) |
| RoPE | theta 10000, YaRN: `factor` 40, `original_max_position_embeddings` 4096, `mscale` 0.707, `mscale_all_dim` 0.707 → 163 840 context |
| Vocabulary | 102 400; `tokenizer.json` (HF BPE), EOS `<｜end▁of▁sentence｜>` |
| Weights | bf16 safetensors, ~31.4 GB |

Two facts shape the whole plan:

1. **The latent cache is 14× smaller than GQA.** 576 elements per token per
   layer against the GQA formula's 8192 — `KvElementsPerTokenPerLayer()`
   already returns this. KV capacity planning changes by an order of
   magnitude, and the latent is replicated (not sharded) under TP.
2. **The checkpoint does not fit the 16 GB development GPU at bf16.**
   Section 18.6 treats serving memory as an explicit decision, not an
   afterthought.

### 18.2 What already exists (verified baseline)

The M9 groundwork means far less is missing than "new architecture" suggests:

| Asset | State |
|---|---|
| `MlaAttentionLayer` (`include/inferx/model/mla.h`, `src/model/mla.cc`) | Complete for one sequence: q down/up projection with the `q_lora_rank == 0` branch (exactly V2-Lite's shape), decoupled-rope split, latent RmsNorm, paged append/gather, unabsorbed attention. Compiled, tested at M9, **called by no model** |
| `MlaAttentionLayer::LayoutFor` | Returns `KvLayout{entries_per_token = 1, kv_heads = 1, head_dim = 576}`; `KvBlockPool` accepts it and `ValueCache()` correctly fails |
| MLA kernels (`csrc/mla.cu`) | Shape-generic — 512/64/128/128 needs no kernel change; bf16 only; attention kernel is the two-pass correctness form |
| `MoeFfn` + `MoeRouteTopK` (`src/model/moe_ffn.cc`, `csrc/moe.cu`) | Softmax-then-top-k with ties to the lower index and optional renormalization is **exactly** V2-Lite's routing (`renormalize = false`); 6 ≤ `kMaxTopK` 16, 64 ≤ `kMaxExperts` 1024; dispatch is stable, atomic-free, graph-capturable |
| Scheduler and KV pool | **Zero changes required.** The scheduler touches only block indices and `block_size`; geometry lives in `KvLayout`, chosen by the model class (T11 held) |
| Tokenizer | `Tokenizer::LoadFrom` probes `tokenizer.json` and loads it through tokenizers-cpp; V2-Lite's BPE artifact and `tokenizer_config.json` EOS resolution work unchanged |
| YaRN machinery | `ComputeYarnInvFreq` + table-driven rope with an attention factor exist — but only on the gpt-oss path, in the full-head form |
| FlashInfer v0.6.9 (vendored) | Exposes `BatchMLAPagedAttention` (FA2, sm_89-usable) and `MLAPlan`, with a `DISPATCH_head_dim` arm for 512 — none of it wired |
| Config MLA fields | `kv_lora_rank`, `q_lora_rank` (tolerates JSON null), `qk_*_head_dim`, `v_head_dim` parsed and validated; V2-Lite passes the MLA validation block |
| M9 test suites | `tests/kernel/mla_test.cc` (722 lines, includes decode-equals-prefill and out-of-order block-table pins) and `moe_test.cc` (737 lines) were deleted by `99e5ed3` with the rest of the old test tree — **recoverable via `git show 99e5ed3^:<path>`** |

### 18.3 Gap analysis

#### Config parsing (`src/model/config.cc`) — four hard failures / silent errors

1. `ParseArchitecture` rejects `DeepseekV2ForCausalLM` (unknown string).
2. `rope_scaling` parsing calls `RequiredString("rope_type")`; V2-Lite spells
   the key **`type`** — a hard `InvalidArgumentError` at load.
3. YaRN `mscale` / `mscale_all_dim` are not parsed and have no `ModelConfig`
   fields; the MLA softmax scale is hardcoded `1/sqrt(nope + rope)` in
   `mla.cc`, wrong for any YaRN-scaled MLA config.
4. MoE keys are spelled differently: InferX reads `num_experts` and
   `shared_expert_intermediate_size`; DeepSeek writes `n_routed_experts` (64)
   and `n_shared_experts` (a **count**, 2 — the width is
   `n_shared_experts × moe_intermediate_size`). Today the file parses with
   `num_experts = 0`, `is_moe()` false, and the model would **silently load
   as dense**. `first_k_dense_replace`, `moe_layer_freq`,
   `routed_scaling_factor`, `scoring_func`, `topk_method` have no fields at
   all.

Also: `q_dim()` / `kv_dim()` and the derived `head_dim` (2048/16 = 128) are
meaningless under MLA and must not be consulted by the DeepSeek path.

#### MLA layer — five gaps

1. **Single-sequence contract.** `MlaAttentionLayer::Forward` takes one
   sequence's flat `block_table` and one scalar `context_len`; `ForwardBatch`
   is a ragged mixed batch. The header explicitly assigns the loop to the
   caller.
2. **No YaRN path.** `MlaRopeInPlace` computes plain `theta^(-2j/rope_dim)`
   frequencies in-kernel; there is no table/attention-factor variant, so
   YaRN frequencies cannot reach the MLA rope at all.
3. **Softmax scale.** `scale = 1/sqrt(192)` must become
   `mscale(mscale_all_dim)² / sqrt(192)` where
   `mscale(m) = 0.1·m·ln(factor) + 1`. For V2-Lite that is
   `(1.2608)² / sqrt(192) ≈ 0.11472`. Note: because
   `mscale == mscale_all_dim`, HF's cos/sin attention factor is exactly 1.0
   for this checkpoint — the entire mscale effect lands in the softmax scale.
4. **RoPE convention unverified.** The kernel uses the half-split (NeoX)
   rotation; HF's DeepSeek code applies `rotate_half` to a de-interleaved
   view. Self-consistent tests cannot catch a mismatch; only real weights
   can. (Resolution in 18.4, D2.)
5. **Performance.** Only the unabsorbed form exists: every step gathers the
   whole context and reconstructs K/V through the `kv_b` GEMM, then runs a
   score-recomputing reference kernel. Correct, unusably slow at real
   context, and its context-dependent scratch shapes block CUDA-graph
   capture.

#### MoE layer — four gaps

1. **Shared-expert gate.** `MoeFfn` *requires* a `shared_gate` weight and
   applies `out += sigmoid(gate)·shared` (Qwen2-MoE's convention). DeepSeek's
   shared experts are an unconditional add. Needs an ungated mode.
2. **Mixed dense/MoE stacks.** Neither existing model mixes layer types;
   nothing in `ModelConfig` expresses "layer 0 is dense". Precedent:
   `layer_is_sliding` for gpt-oss.
3. **Expert weight stacking.** The checkpoint stores
   `mlp.experts.{0..63}.{gate,up,down}_proj.weight` as ~192 tensors per
   layer; `MoeWeights` wants stacked `[E, 2·inter, hidden]` / `[E, hidden,
   inter]`. No stacking upload helper exists (`UploadConcatenated` is 2-D
   only).
4. **The bf16 expert path has never met a real model** — it runs 64 cuBLASLt
   calls per layer behind a host readback of dispatch offsets each step,
   which also makes it uncapturable as a graph. Acceptable for bring-up;
   a performance-phase item, not a correctness one.

#### Model class and serving wiring

- No `DeepseekV2Model`; no loader references any MLA weight name.
- `Engine::Create` dispatches on a single `if (arch == kGptOss)`; the
  else-arm runner (`QwenRunner`) is a clean 7-method interface but lexically
  bound to `model::Qwen2Model` throughout.
- The chat template is one hard-coded Qwen2 ChatML function selected
  unconditionally in `tokenization_service.cc`; V2-Lite's template is not
  ChatML (`<｜begin▁of▁sentence｜>{system}` … `User: …\n\nAssistant: …<｜end▁of▁sentence｜>`),
  and emitting `<|im_start|>` tokens at a DeepSeek vocabulary would be
  silently wrong.
- No model, kernel, config, or scheduler tests remain in the tree (all
  deleted by `99e5ed3`); the DeepSeek work needs them restored first.

### 18.4 Design decisions

**D1 — Keep the fused 576-wide latent cache row.** `MlaAppendLatent` writes
`[latent | rope_key]` as one row and the M9 paging tests pin that layout.
FlashInfer's `MLAParams` takes separate `ckv`/`kpe` pointers *with separate
strides*, so the fused row can be presented as two strided views of the same
storage (`ckv = base, kpe = base + 512`, both with row stride 576) when the
FlashInfer wrapper lands. Do not split the cache; verify the strided-view
route in the performance phase and split only if FlashInfer's plan rejects
it.

**D2 — Settle the RoPE convention in the loader, not the kernel.** Both rope
inputs are produced by projections whose *output channels* the loader owns:
`q_proj`'s trailing 64 columns per head and `kv_a_proj_with_mqa`'s trailing
64 rows. If HF's interleaved convention differs from the kernel's half-split
form, permute those weight rows once at load so the kernel's convention
becomes correct by construction — the cached rope key is then consistently
permuted, and only dot products against equally-permuted queries matter.
The golden-logits test (D5 phase) is the arbiter; the permutation is a
loader constant, not a runtime branch.

**D3 — Extend `MoeFfn` minimally: `shared_gated` flag and
`routed_scaling_factor`.** Ungated mode reuses `AddInPlace` (or a trivial
kernel) instead of `MoeAddSharedExpert`, and `shared_gate` becomes optional
exactly when `shared_gated == false`. `routed_scaling_factor` multiplies the
combine weights (1.0 for V2-Lite — plumbed but inert, so V3-family configs
are not silently wrong later). Reject `scoring_func != "softmax"` and
`topk_method != "greedy"` at config validation with typed errors.

**D4 — Per-layer FFN kind on `ModelConfig`.** Add `first_k_dense_replace`
and `moe_layer_freq` plus an `IsMoeLayer(i)` predicate (the
`IsSlidingLayer` pattern). The model class branches per layer between the
dense FFN weights (`mlp.{gate,up,down}_proj`) and `MoeFfn`.

**D5 — MLA YaRN via a table variant.** Add `MlaRopeFromTable(x, rope_dim,
positions, inv_freq[rope_dim/2], attn_factor)` beside `MlaRopeInPlace`, and
generalize the host-side attention-factor computation so DeepSeek's
`0.1·mscale·ln(factor) + 1` form and gpt-oss's default coexist. The softmax
scale becomes a parameter of `MlaAttentionLayer::Create` (the header already
anticipates this), computed once from config: for V2-Lite ≈ 0.11472.

**D6 — Batched serving by a per-sequence loop first.** `DeepseekV2Model`
iterates sequences of the `ForwardBatch`, slicing the flat token arrays and
the sequence's `block_table` row per call — the loop the MLA header assigns
to the caller. This is O(Σ context) reconstruction per layer per step and is
accepted for bring-up; the performance phase replaces the attention inner
loop, not the model structure. The executor must honor `batch.decode_only`
(the uncommitted scheduler/`ForwardBatch` fix) rather than inferring decode
from shape — it will be the first executor to do so.

**D7 — Engine integration by de-Qwening the runner.** Rename `QwenRunner`
to `ModelRunner` and make `SingleQwenRunner` a `SingleModelRunner<TModel>`
template over the existing duck-typed 7-method contract, rather than adding
a third parallel serving loop in `engine.cc`. The NCCL variant stays
Qwen2-only (MLA TP is out of scope; the latent replicates rather than
shards, and the engine already rejects TP > 1 for non-Qwen2). Bring-up may
start on the simpler synchronous `RunGptOss` loop shape, but the target is
the async `StepAsync`/`AwaitStep` path with device sampling so chat serving
gets temperature/top-p, not host argmax.

**D8 — Chat template as a selected kind, not a sniffed string.** Transcribe
`ApplyDeepSeekV2ChatTemplate` beside the Qwen2 function (same
tested-transcription policy; a Jinja engine remains out of scope) and add a
`ChatTemplateKind` chosen at composition time — the server already knows the
architecture where it constructs `InProcessTokenizationService`. This is
the section 17.8 seam ("chat templates selected by immutable model
version"), delivered early because DeepSeek forces it.

### 18.5 HTTP-server integration details

The section 17 gateway architecture needs no structural change — that is the
point of the seams. Specific touch points:

- **Model registry entry.** `id` = directory basename or
  `--served-model-name` as today; capabilities must carry the real limits:
  context length (advertise the practical serving limit, not 163 840, until
  long-context YaRN serving is validated), no embeddings, streaming yes.
  `GET /v1/models` then reports it without code changes.
- **Tokenization service.** `TokenizeChat` routes through the
  `ChatTemplateKind` (D8); `TokenizeCompletion` is unchanged.
  `POST /v1/tokenize` works once the tokenizer loads; the known
  `add_special_tokens = true` limitation (post-processor tokens
  unimplemented) applies to DeepSeek exactly as to Qwen and keeps its
  existing typed error.
- **Validation and admission.** `max_tokens`/context checks read the model's
  registry metadata — no DeepSeek constants in handlers. The admission KV
  reservation estimate uses `KvElementsPerTokenPerLayer()`, which is already
  MLA-correct; capacity numbers simply improve 14×.
- **Sampling parameters.** Recommended defaults (e.g. temperature 0.3 for
  V2-Lite-Chat) are a client concern; the server validates ranges as in
  section 9, unchanged.
- **Remote scheduler gateway.** The gateway loads only a tokenizer directory
  and a model id — DeepSeek works there once the tokenizer directory and the
  template kind are supplied; the gateway needs the same `ChatTemplateKind`
  selection switch, driven by configuration rather than a loaded checkpoint.

### 18.6 Serving memory on the 16 GB development GPU

bf16 weights are ~31.4 GB; the box has 16 GB (sm_89). The weight mass is
almost entirely routed experts: 64 × 3 × 1408 × 2048 × 26 layers ≈ 14.4 B of
the 15.7 B parameters. Options, in preference order:

1. **W4A16 routed experts, bf16 everything else.** ~7.2 GB experts (+ ~0.2 GB
   group scales) + ~2.9 GB for attention/dense/shared/embeddings ≈ 10.5 GB,
   leaving several GB for the latent KV pool — which at 31 104 bytes per
   token (576 × 2 B × 27 layers) buys ~34 000 cached tokens per GB. The
   `W4A16Gemm` kernel and int4 quantization kernels exist; what does not is
   an int4 *expert* path in `MoeFfn` (a per-expert loop mirroring the bf16
   loop is the bring-up form; a grouped int4 GEMM is a performance item).
2. **Rented GPU at bf16** (the TP-validation pattern): correct first serving
   numbers without new quantization code, at rental cost and iteration
   latency.
3. FP8 weights (~15.7 GB) do not leave room for a KV pool on this box; not
   viable.

Correctness does not wait on this choice: layer-level tests run at toy
shapes on-box, and golden logits come from HF on CPU (one-off, RAM
permitting) or from the rented-GPU run. The empty
`bench-results/deepseek-v2-lite-vllm-20260808-1535/` directory suggests a
vLLM baseline attempt already started; rerun it wherever the serving
hardware decision lands.

### 18.7 Phased implementation plan

Order matters: the tree currently has the `src/kernels/` → `csrc/` rename
half-landed (deletions staged, `csrc/` untracked) and the
`ForwardBatch::decode_only` fix uncommitted. Land those first — every phase
below touches files near them.

#### Phase D0: land the in-flight rename and restore the test floor

- Commit the `csrc/` move and the `decode_only` scheduler/`ForwardBatch`
  change as separate commits.
- Restore from `99e5ed3^`: `tests/kernel/mla_test.cc`, `moe_test.cc`,
  `config_test.cc`, `scheduler_test.cc`, `prefix_cache_test.cc`,
  `safetensors_test.cc`, and the old `tests/CMakeLists.txt` harness
  (`inferx_add_kernel_test`, `kernel` labels, GPU self-skip). Target names
  survived the rename (`inferx::kernels` is unchanged), so link lines
  resolve.
- Reconfigure the stale `build/` tree (it predates the rename).

Exit: `ctest -L unit` is green host-only; `ctest -L kernel` is green on the
box; the M9 MLA/MoE pins (decode-equals-prefill, out-of-order block table,
router determinism) all run again.

#### Phase D1: config parsing (host-only)

- Add `Architecture::kDeepSeekV2` ↔ `"DeepseekV2ForCausalLM"`.
- Accept `rope_scaling.type` as an alias for `rope_type`; parse `mscale` and
  `mscale_all_dim` into new YaRN fields.
- Alias `n_routed_experts` → `num_experts`; derive
  `shared_expert_intermediate_size = n_shared_experts × moe_intermediate_size`;
  parse `first_k_dense_replace`, `moe_layer_freq`, `routed_scaling_factor`,
  `scoring_func`, `topk_method`; add `IsMoeLayer(i)`.
- Validation arm for `kDeepSeekV2`: MLA fields required, reject non-softmax
  scoring / non-greedy top-k with typed `Unimplemented` errors, forbid
  consulting `q_dim()`/`kv_dim()` under MLA (assert or refactor call sites).
- Tests: a checked-in V2-Lite `config.json` fixture parses to exact field
  values; every alias and every rejection path is covered. This phase alone
  removes all four silent-failure modes of 18.3.

Exit: the real `config.json` round-trips; host-only tests cover the aliases,
`type` vs `rope_type`, `q_lora_rank: null`, and `is_moe()` truth.

#### Phase D2: MoE generalization

- `MoeFfn::Config` gains `shared_gated` (default true — existing behavior)
  and `routed_scaling_factor` (default 1.0). Ungated mode makes
  `shared_gate` optional and adds the shared output with `AddInPlace`.
- Extend the restored `moe_test.cc`: ungated shared expert changes every row
  by exactly the shared-MLP output; scaling factor multiplies combine
  weights; gated behavior unchanged.

Exit: both shared-expert conventions tested against host references; the
existing gpt-oss path is bit-identical.

#### Phase D3: MLA YaRN and softmax scale

- `MlaRopeFromTable` kernel variant (trailing-slice, table + attention
  factor); generalized YaRN host computation with the DeepSeek mscale form.
- `MlaAttentionLayer::Create` takes the softmax scale (config-computed);
  the two rope call sites switch to the table variant when YaRN is active.
- Tests: table variant equals `MlaRopeInPlace` when the table is unscaled
  and the factor is 1; the V2-Lite scale constant (≈ 0.11472) is pinned;
  frequencies match `_compute_yarn_parameters` for the V2-Lite parameters.

Exit: MLA layer tests pass with and without YaRN; no change to non-MLA
paths.

#### Phase D4: `DeepseekV2Model`

New `src/model/deepseek_v2.{h,cc}` following the `GptOssModel` structure
(pimpl, non-blocking stream, `PrepareBatchInputs` / `RunPagedForward` split,
grow-only scratch):

- Loader: `q_proj` (no `q_a` — the `q_lora_rank == 0` branch),
  `kv_a_proj_with_mqa`, `kv_a_layernorm`, `kv_b_proj`, `o_proj`; per-layer
  dense-vs-MoE branch; expert stacking upload (one batched staging copy per
  stacked tensor, not 192 `cudaMemcpy`s per layer); fused shared-expert MLP
  into `shared_gate_up`/`shared_down`; the D2 loader-side rope permutation
  if the golden test demands it.
- `AttachKvCache` uses `MlaAttentionLayer::LayoutFor(config)`.
- `Step`: per-sequence MLA loop over the ragged batch (D6), honoring
  `batch.decode_only`; `Forward(tokens, out_logits)` single-prompt reference
  path for the logits harness.
- CUDA graphs: **explicitly deferred** — the unabsorbed MLA scratch shapes
  depend on context length; `CaptureDecodeGraph` returns unimplemented and
  the engine skips capture for this architecture.
- Tests: `deepseek_v2_model_test.cc` modeled on `gpt_oss_model_test.cc` at
  toy shapes (host-reference forward, decode-equals-prefill through the
  paged cache, dense layer 0 vs MoE layer 1 branch).

Exit: toy-shape model forward matches a host reference; paged decode equals
prefill; the loader maps every tensor name in the real checkpoint index
(verifiable host-only against the safetensors header without weights).

#### Phase D5: engine, runner, template, serving

- Rename `QwenRunner` → `ModelRunner`; `SingleModelRunner<TModel>` template;
  `engine.cc` grows a `kDeepSeekV2` arm that builds it (TP > 1 stays
  rejected).
- `ApplyDeepSeekV2ChatTemplate` + `ChatTemplateKind` selection at
  composition (18.5); template unit tests against fixtures generated by HF's
  `apply_chat_template` for the same conversations, including the
  no-system-message and multi-turn cases.
- Golden logits: adapt `scripts/gen_reference_logits.py`; generate on CPU or
  rented GPU (18.6); wire the logits-comparison test.
- Serve end-to-end on the chosen memory strategy; capture the section 17.3
  compatibility fixtures for the new model id.

Exit: `inferx-serve --model <deepseek-dir>` answers `/v1/chat/completions`
(streaming and blocking) with template-correct prompts; logits match HF
within the established tolerance; the RoPE-convention question is settled by
measurement and documented in ARCHITECTURE.md.

#### Phase D6: performance (separately gated, after correctness)

In expected-value order:

1. **FlashInfer MLA decode** via `BatchMLAPagedAttention`
   (`HEAD_DIM_CKV` 512, `HEAD_DIM_KPE` 64, FA2 path, sm_89) over the fused
   cache row presented as strided `ckv`/`kpe` views (18.4 D1) — replaces
   the gather + reconstruct + reference-attention chain for decode, and is
   the "FlashInfer MLA wrapper" item DEVELOPMENT.md already names as the
   reason for the CUDA 13 floor. Absorption (folding `kv_b` into Q/O) is the
   alternative if the wrapper fights back; it changes no interfaces.
2. **bf16 grouped expert GEMM** (CUTLASS grouped GEMM, or extend the
   `Mxfp4GroupedGemm` device-dispatch pattern) — removes 64 launches and one
   host sync per MoE layer per step, and unblocks graph capture of the FFN.
3. **CUDA graphs for decode** once (1) and (2) give fixed shapes; honor
   `decode_only`.
4. **Device sampling / `StepAsync`** parity with the Qwen2 path.
5. If serving on-box: the int4 expert path (18.6), first as a per-expert
   loop, then grouped.

Each item lands behind the D5 correctness fixtures re-run; no perf change
merges on a red golden-logits test.

### 18.8 Risks and open questions

| Risk | Mitigation |
|---|---|
| RoPE interleave mismatch produces fluent-but-wrong output that layer tests cannot catch | The D5 golden-logits gate is mandatory before any serving claim; 18.4 D2's loader permutation is the prepared fix |
| YaRN correctness beyond 4096 context is untested even with matching logits at short context | Advertise the validated context in the registry; extend with a long-context perplexity check before raising it |
| bf16 stacked-expert path is unexercised at real scale (64 experts × 26 layers) | Phase D2/D4 toy tests first; watch the per-layer host sync cost in bring-up profiling, already slated for removal in D6 |
| FlashInfer `MLAParams` may not accept the fused-row strided views | Verified in a spike at the start of D6; fallback is a cache split behind `LayoutFor` (mechanical) or the absorbed-form kernels |
| 16 GB box cannot hold bf16 weights | Decided in 18.6 before D5 serving work; correctness phases are unaffected |
| `Qwen2MoeForCausalLM` precedent: enum arms without implementations rot silently | D1 makes unsupported combinations hard errors; the config fixture tests pin every accepted and rejected field |

The recommended delivery units are D0 through D5, in order; D6 items land
individually afterwards. D0 and D1 are pure debt-paydown and are worth
landing even if DeepSeek work pauses: they restore the deleted M9 test
assets and make config parsing honest about what it does not support.
