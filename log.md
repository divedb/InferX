# InferX logging design

Status: proposal, pre-implementation.

## 1. Third-party investigation

Surveyed `third_party/` for a reusable logging library:

| Candidate | Verdict |
|---|---|
| `abseil-cpp` — `absl/log/` | **Use it.** Checked out, and already a mandatory build dependency (`add_subdirectory` in `cmake/InferXDependencies.cmake`; `absl::status/strings/...` are linked by `inferx::core` and `inferx::support` today). Ships full logging: `LOG()`, `VLOG()`, `CHECK()`, severity globals, pluggable `absl::LogSink`, CLI flags (`absl/log/flags.h`), and `absl::ScopedMockLog` for tests. |
| `folly` — `folly/logging/` | Rejected. Folly is an optional submodule (`update = none` in `.gitmodules`, gated in `InferXDependencies.cmake`); the core engine cannot depend on it. |
| `boost` (Boost.Log) | Rejected. Also optional (`update = none`), checked out only for Beast headers; Boost.Log's compiled libs aren't built. |
| others (cutlass, flashinfer, mimalloc, cpp-httplib, prometheus, grafana, tokenizers-cpp, googletest, fast_float) | Not logging libraries. `prometheus`/`grafana` are for metrics dashboards and stay complementary to logs. |

**Decision: build on `absl/log`.** Zero new dependencies, one implementation shared by every target, and it composes with the `absl::Status` error plumbing the codebase already uses (`engine.status().ToString()` etc.).

## 2. Current state

~23 ad-hoc `std::fprintf(stderr, ...)` / iostream call sites, concentrated in:

- `src/main/main.cc`, `src/server/gateway/main.cc` — CLI errors, startup/shutdown banners.
- `src/model/deepseek_v2.cc`, `src/model/gpt_oss.cc` — weight-load progress and diagnostics.
- `src/engine/engine.cc` — `engine step failed: ...` (the only runtime error path that logs today).

Problems: no severity levels, no timestamps, no thread/source location, no request correlation, no runtime verbosity control, and load-progress spam can't be turned off.

## 3. What to record

### 3.1 Severity ladder

- `FATAL` (via `CHECK`/`LOG(FATAL)`) — programming invariants only (shape mismatches, impossible enum values). Never for bad user input or recoverable I/O.
- `ERROR` — an operation failed and a request or the process is affected: engine step failure, model load failure, listen failure, CUDA errors.
- `WARNING` — degraded but continuing: KV-cache pressure/preemption, slow weight mmap, retryable transport errors, config fallbacks.
- `INFO` — low-volume lifecycle events (see 3.2). Steady-state serving must be near-silent at INFO except the access log.
- `VLOG(n)` — debug tracing, compiled in but off by default:
  - `VLOG(1)` per-request lifecycle detail (state-machine transitions in `src/server/request/state_machine.cc`, admission decisions, scheduler placement).
  - `VLOG(2)` per-step detail (batch composition, token counts per step, KV block alloc/free).
  - `VLOG(3)` per-layer/tensor detail (the current per-layer load prints move here).

### 3.2 Event categories

1. **Process lifecycle** (INFO): version/build info, parsed config (model dir, devices, quantization mode, ports), "listening on ...", shutdown reason. One line each.
2. **Model loading** (INFO summary, VLOG(3) per-layer): loader start with model type + dir, total bytes/tensors, elapsed time, device memory after load. The existing per-layer progress lines in `deepseek_v2.cc` become `VLOG(3)`.
3. **Request access log** (INFO, one line per finished request): `request_id`, route, model, HTTP status, prompt/completion token counts, queue time, TTFT, total latency, finish reason. Emitted at request completion in the request service — this is the one steady-state INFO stream.
4. **Scheduler/admission** (WARNING on rejection/preemption, VLOG(1) on decisions): admission rejects with reason, queue depth at reject, preemptions/evictions.
5. **Engine/runtime errors** (ERROR): step failures with `absl::Status`, CUDA/OOM errors with device id, watchdog events.
6. **Auth/gateway** (WARNING): auth failures (key id only — never the key), upstream connect failures.

### 3.3 Redaction rule

Never log prompt or completion text, API keys, or full request bodies at INFO or above. Token *counts* are fine. Content may appear only under a dedicated `VLOG(4)` debug level that is documented as unsafe for production.

## 4. Design

### 4.1 Wrapper: `inferx::support` gains a logging header

New files:

- `include/inferx/support/log.h` — includes `absl/log/log.h` + `absl/log/check.h` (single include for call sites) and declares:
  - `void InitLogging(const LogOptions&);` — process-wide init for the binaries.
  - `struct LogOptions { std::string min_level = "info"; int verbosity = 0; bool json = false; std::optional<std::string> file; };`
- `src/support/log.cc` — implementation: `absl::InitializeLog()`, `absl::SetStderrThreshold(...)`, `absl::SetGlobalVLogLevel(...)`, sink registration.
- `src/support/CMakeLists.txt` — add `log.cc`; link `absl::log`, `absl::log_initialize`, `absl::log_globals`, `absl::log_sink_registry` (PUBLIC, so every `inferx::*` target that already links `inferx::support` gets logging transitively).

Rationale for placement: `src/support` is the "small utilities, no device or model dependencies" layer that everything already links; `src/observe` stays metrics-only.

Call sites use plain `LOG(INFO)` / `VLOG(1)` / `CHECK(...)` from absl — no bespoke macros to learn. The wrapper only owns initialization, sinks, and helpers.

### 4.2 Sinks and formats

- **Default (dev):** absl's stderr sink. Format: `I0809 14:22:31.123456 tid file.cc:123] message` — free.
- **Production (`json = true`):** a custom `absl::LogSink` (`src/support/log.cc`) writing one JSON object per line to stderr: `{"ts","severity","file","line","tid","msg"}`, using the existing `inferx/support/json.h` helpers. When enabled, the default stderr sink is suppressed via `absl::SetStderrThreshold(kInfinity)` so lines aren't doubled.
- **Optional file sink** (`file` option): same format as chosen above, appended to a path; rotation is delegated to the operator (logrotate/container runtime), not implemented.

### 4.3 Request correlation

absl/log has no MDC/context field, so correlation is by convention:

- The access-log line and any per-request LOG/VLOG must start with the request id: `LOG(INFO) << Rid(id) << "..."` where `Rid` is a tiny formatter in `log.h` producing `[req_...] `.
- `request_id` already exists (`src/server/request/request_id.cc`, UUIDv7-style `req_` prefix) and is threaded through handlers/request_service/scheduler_client — no new plumbing needed, just discipline at call sites. Grep-ability (`grep req_0198...`) is the goal; structured per-field JSON for the access log can come later by giving the access log its own sink if needed.

### 4.4 CLI flags

Both binaries (`inferx-server`, `inferx-gateway`) use hand-rolled arg parsing, so we add options there rather than adopting `absl::ParseCommandLine`:

- `--log-level={debug,info,warning,error}` → stderr threshold (debug maps to INFO threshold + verbosity 1).
- `--v=N` → global VLOG level.
- `--log-json` → JSON sink.
- `--log-file=PATH` → additional file sink.

Defaults preserve today's behavior (human-readable stderr, INFO).

### 4.5 Failure semantics

- `LOG(FATAL)`/`CHECK` abort with a stack trace — keep enabled in release builds; invariant violations in an inference engine corrupt outputs silently otherwise.
- Logging must never throw and must be safe before `InitLogging` (absl guarantees both; pre-init lines go to stderr).

## 5. Migration plan

Phased; each phase builds green and is independently landable.

1. **Foundation.** Add `log.h`/`log.cc`, CMake wiring, `InitLogging` called from both `main.cc`s behind the new flags. Add a unit test using `absl::ScopedMockLog` (googletest is already a dependency) verifying init, JSON sink output shape, and `Rid` formatting.
2. **Convert error paths.** Replace the `fprintf` calls in `src/engine/engine.cc`, `src/main/main.cc`, `src/server/gateway/main.cc` with `LOG(ERROR)`/`LOG(INFO)`. CLI *usage* errors (bad flags, `--help`) stay on raw `fprintf` — that's UI, not logging.
3. **Convert model loaders.** `deepseek_v2.cc`, `gpt_oss.cc`: summary → `LOG(INFO)`, per-layer progress → `VLOG(3)`. This also fixes the load-spam problem.
4. **Access log.** One `LOG(INFO)` line at request completion in the request service with the fields from §3.2.3, prefixed with `Rid(id)`.
5. **Scheduler/admission + VLOG tracing.** Add WARNING on rejects/preemptions and `VLOG(1..2)` tracing per §3.1.
6. **Enforcement.** Once phases 2–3 land, add a CI grep (or clang-tidy check) rejecting new `std::cout|std::cerr|fprintf(stderr` in `src/` outside the arg-parsing blocks.

Out of scope for now: log rotation, shipping to a collector (operators handle transport; JSON-lines makes that trivial), OpenTelemetry trace propagation (revisit if the gateway→server hop needs distributed tracing).

## 6. Open questions

- Should the access log be its own sink/stream (e.g. separate fd) instead of interleaved INFO? Deferred — start interleaved, split later if operators ask.
- `--log-level=debug` mapping to `--v=1` is a convenience alias; exact VLOG level assignments per module can be tuned during phase 5.
