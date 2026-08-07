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

## 17. Implementation sequence and success criteria

Recommended implementation order:

1. Pin Boost and Folly versions and add narrow CMake targets.
2. Use Beast request/response messages directly, define the inference-specific
   coroutine response writer, and adopt Folly cancellation tokens.
3. Implement and race-test the Beast/Folly awaitable adapter.
4. Add the Asio listener and per-connection Beast session.
5. Move health and model-list routes to coroutine handlers.
6. Introduce `RequestContext` and its coroutine state machine.
7. Move non-streaming completion handling to the new server.
8. Add bounded event buffers, SSE writes, and disconnect cancellation.
9. Migrate streaming completion handling and remove `cpp-httplib`.
10. Add authentication, quotas, admission, and distributed tracing.
11. Define the scheduler Protobuf boundary when process separation is needed.
12. Add model registry and lifecycle control-plane handlers.

The design is successful when:

1. HTTP handlers contain no tokenizer, scheduler, or GPU-specific logic.
2. Blocking and streaming generation consume the same event abstraction.
3. Chat-template selection is model-version-aware.
4. Protocol code remains testable without CUDA or sockets.
5. Completion-service tests can use fake scheduler and tokenizer clients.
6. Existing OpenAI wire behavior remains compatible.
7. Client disconnects promptly cancel execution and release KV cache.
8. Gateways can scale horizontally without owning model execution state.
9. No synchronous or blocking wait runs on an I/O executor.
10. Cancellation-versus-completion races pass deterministic stress tests and
    sanitizers.
11. The old `cpp-httplib` dependency and thread-per-connection implementation
    are removed after wire-compatibility tests pass.
