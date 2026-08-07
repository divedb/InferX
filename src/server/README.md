# HTTP server modules

This directory follows the code-level design in [`http_server.md`](../../http_server.md).
The module boundaries keep external protocol handling, request lifecycle,
admission, scheduler communication, and operational concerns separate from
model execution.

| Directory | Responsibility |
|---|---|
| `api/` | OpenAI request/response types, parsing, encoding, and error mapping |
| `transport/` | Asio/Beast runtime, sessions, routing, SSE writes, and the private Folly adapter |
| `coroutine/` | Deadline, cancellation, executor-hop, and structured-task helpers |
| `middleware/` | Request IDs, authentication/authorization, rate limiting, and tracing |
| `request/` | Externally visible request context, lifecycle, state, and cancellation |
| `admission/` | Quota and capacity admission policy |
| `tokenization/` | Server-side tokenizer selection, chat templates, and token accounting |
| `streaming/` | Bounded event delivery, routing, and slow-consumer backpressure |
| `scheduler_client/` | Scheduler submission, event subscription, cancellation, and retry policy |
| `model_registry/` | Model/version discovery, routing, and lifecycle clients |
| `auth/` | Principals, API-key storage contracts, and RBAC |
| `observability/` | HTTP-server metrics, structured logging, and tracing |
| `admin/` | Model/system administrative handlers and graceful shutdown |

Public contracts belong under `include/inferx/server/<module>/`; private
implementations belong here. Add source files to `src/server/CMakeLists.txt` as
each module is implemented. The existing `transport/` module is host-only and
is built independently so protocol and coroutine adapter tests do not require
CUDA.
