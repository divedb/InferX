# Logging

How to log from InferX code and how to configure it at startup. The API is
`include/inferx/core/logging.h`, the implementation is `src/core/logging.cc`,
and the behavior described here is pinned by `tests/unit/core/logging_test.cc`.

InferX logging is a thin, deliberate layer over Abseil logging: the macros
forward to `ABSL_LOG`/`ABSL_VLOG`/`ABSL_CHECK`, and initialization wires the
process-wide Abseil globals from one `LoggingConfig`. Code logs through the
`INFERX_*` macros only — never raw Abseil macros — so the surface stays
greppable and swappable.

---

## 1. Logging from code

### 1.1 Severity levels: `INFERX_LOG`

```cpp
INFERX_LOG(INFO) << "mapped " << shards << " shard(s)";
INFERX_LOG(WARNING) << "kv pool at " << used << "/" << capacity;
INFERX_LOG(ERROR) << "batch " << id << " failed: " << status;
```

Streams like `std::ostream`. Severities:

| Severity | Use for | Filtered by |
|---|---|---|
| `INFO` | One-per-step milestones, startup facts | `min_level` |
| `WARNING` | Recovered-from or degraded operation | `min_level` |
| `ERROR` | A failed operation the caller must see | `min_level` |
| `FATAL` | Unrecoverable invariant violation | never; logs then terminates |

`FATAL` is not a configurable level: it always reaches the log and terminates
the process. Do not use it for recoverable errors — return a `Status`.

### 1.2 Verbosity tracing: `INFERX_VLOG`

```cpp
INFERX_VLOG(1) << "step " << step << ": " << tokens << " tokens";
INFERX_VLOG(3) << "kernel " << name << ": " << micros << " us";
```

A `VLOG(n)` site emits only when the effective verbosity for its source file is
`>= n`. The effective verbosity is the global `vlog_verbosity`, overridden by the
first matching pattern in `vlog_modules`. Recommended levels:

* **1** — per-request or per-step milestones.
* **2** — per-step detail (shapes, cache hit rates, scheduling decisions).
* **3+** — hot-path tracing; expect large volumes.

A disabled site costs a check of a cached verbosity, so `VLOG` may live on
per-token paths. Module patterns match the source file's basename and support
`*` (e.g. `model_runner*` covers `model_runner.cc`).

**`VLOG` is emitted at INFO severity**, so `min_level` above `info` hides all
verbose output too, including module overrides.

### 1.3 Invariants: `INFERX_CHECK` / `INFERX_DCHECK`

```cpp
INFERX_CHECK(shard != nullptr) << "missing shard " << index;
INFERX_DCHECK(row < rows);  // Compiles away in release (NDEBUG) builds.
```

Both abort with a message when the condition is false. Use `CHECK` for
invariants whose violation makes all bets off; use `DCHECK` for expensive or
development-only assertions. Prefer returning a `Status` for expected failures
(bad input, exhausted resources) — checks are not error handling.

---

## 2. Configuration

`LoggingConfig` and its loader never read `argv` — the application owns flag
parsing and applies flags on top of the loaded config. The environment loader
layers, lowest to highest precedence:

1. `LoggingConfig` defaults (`info`, verbosity 0, stderr only).
2. The `logging:` section of the YAML file named by `INFERX_CONFIG_FILE`.
3. `INFERX_LOG_LEVEL` and `INFERX_VLOG`.

```yaml
# Everything under logging: is optional.
logging:
  level: info            # info | warning | error
  vlog: 1                # global verbosity, >= 0
  vlog_modules:          # per-file overrides, glob patterns allowed
    model_runner*: 2
    scheduler*: 3
  file: /var/log/inferx/serve.log   # written alongside stderr
```

Other top-level YAML sections (`serve:`, `bench:`, ...) belong to other
components and are ignored by the logging loader.

| Variable | Meaning | Invalid value |
|---|---|---|
| `INFERX_CONFIG_FILE` | Path to the YAML config file | Missing or unparsable file is a startup error |
| `INFERX_LOG_LEVEL` | `info` \| `warning` \| `error` (lowercase) | Startup error |
| `INFERX_VLOG` | Non-negative integer | Startup error |

**Bad configuration fails loudly.** A typo in a variable, an unknown `logging:`
key, or an unparsable file returns an error `Status` naming the problem — it
never silently falls back to defaults, so a misconfigured deployment cannot
quietly lose its logs.

---

## 3. Initialization

```cpp
#include "inferx/core/logging.h"

int main(int argc, char** argv) {
  auto config = inferx::core::LoggingConfigFromEnvironment();
  if (!config.ok()) {
    std::cerr << config.status() << "\n";  // Logging is not up yet; say it ourselves.
    return 1;
  }
  // Apply CLI flags to *config here, if the app has any.
  if (!inferx::core::InitializeLogging(*config).ok()) return 1;
  // ... worker threads may spawn from here on
}
```

The contract of `InitializeLogging`:

* **Call it exactly once**, at the top of `main`, after config/CLI parsing and
  before worker threads spawn. A second call returns
  `FailedPreconditionError` rather than reconfiguring sinks.
* **Rejection is side-effect free.** Validation (negative verbosities, a
  `min_level` above `error`, an unopenable `log_file`) happens before any
  global state changes, so a rejected call leaves logging untouched.
* **`min_level` is what you see.** Both the sink threshold and the stderr
  threshold are set to it.
* **The file sink appends** and is deliberately never removed, so logging
  during static destruction cannot touch a dead sink. Entries at `WARNING` or
  above are flushed immediately; `INFO` is buffered.

Output lines follow Abseil's format — severity letter, timestamp, thread id,
source location:

```
I1012 10:47:51.850989 145592 logging_test.cc:242] plain info line
W1012 10:47:51.851035 145592 logging_test.cc:246] flush marker
```
