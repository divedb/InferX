# InferX observability design

Status: implementation plan for M7 tensor parallelism. Milestones 1 through 4
are implemented, including opt-in sampled CUDA-event timing with a preallocated
per-rank ring. Completed events are polled on later collective calls without a
stream/device synchronization; graph capture is detected and skipped.

## Scope and reviewed implementations

The initial deployment is one machine with two GPUs and NCCL. The metric model must nevertheless use a rank/world-size model rather than encode “GPU 0/GPU 1”, so it works for more local GPUs and future communicator backends such as MSCCL++.

This proposal was informed by these local revisions:

- vLLM `fd47e57f4b0d` (`v0.6.3`): `vllm/engine/metrics.py` and `docs/source/serving/metrics.rst`.
- SGLang `b8ec544946f1`: `python/sglang/srt/observability/metrics_collector.py` and `experimental/sgl-router/src/server/metrics.rs`.
- InferX: `include/inferx/server/engine.h` and the existing `/stats` handler in `src/server/http_server.cc`.

vLLM establishes the essential serving signals: running/waiting/swapped requests, KV-cache occupancy, prefix-cache hits, preemptions, prompt and generated token throughput, time to first token (TTFT), time per output token, end-to-end latency, request lengths, and finish reason. SGLang adds useful operational detail: queue time, inter-token latency (ITL), absolute token-pool capacity, retractions, CUDA graph mode, batch occupancy, forward cost, startup/model-load information, aborts, and transfer metrics for disaggregated serving. Its router implementation also demonstrates that exposition need not put a third-party client library in the request path.

InferX already owns cumulative step/token/preemption/cache counters plus running/waiting request and block state. Those values should become the first metrics rather than building a parallel accounting system.

## Architecture

```text
requests -> InferX HTTP server -> scheduler/model workers -> communicator (NCCL)
                 |                       |                    |
                 +---- /metrics <--------+---- metric events--+
                           ^
                           | scrape
                    Prometheus server <----- NVIDIA DCGM exporter
                           ^                  (GPU/NVLink/PCIe)
                           |
                        Grafana
```

- InferX exposes Prometheus text format at `GET /metrics` on its existing HTTP listener. `/stats` remains available for compatibility and debugging.
- Prometheus scrapes InferX and NVIDIA DCGM Exporter. Node Exporter is optional for CPU, memory, disk, and network correlation.
- Grafana reads only from Prometheus and is provisioned with version-controlled data-source and dashboard files.
- `third_party/prometheus` and `third_party/grafana` pin the monitoring services for reproducibility. They are deployment/tooling dependencies and are not linked into the InferX C++ binary. Normal deployments should use matching tagged container images rather than compile these large Go projects.
- InferX will initially use a small internal registry and standards-compliant text exposition. This keeps the hot path dependency-free and lets the same instrumentation survive a future monitoring-client change.
- Production/cloud deployments must bind monitoring endpoints to localhost or a private network and place authentication/TLS at the ingress; `/metrics` must not be exposed publicly.

## Metric contract

All public names start with `inferx_`, use base units (`_seconds`, `_bytes`), and include HELP and TYPE declarations. Counters are process-lifetime monotonic values. Ratios are reported as 0–1. Histograms export cumulative `le` buckets, `_sum`, and `_count`.

Allowed labels are bounded: `model`, `endpoint`, `outcome`, `finish_reason`, `phase` (`prefill`/`decode`), `rank`, `backend`, `op`, `graph_mode`, and `precision`. A metric should use only the labels it needs. Request IDs, prompts, clients, sequence lengths, batch IDs, and error strings are forbidden labels. Build/model metadata belongs in a single `inferx_engine_info` gauge.

### Serving and scheduler

| Metric | Type | Purpose |
|---|---|---|
| `inferx_requests_total{endpoint,outcome,finish_reason}` | counter | Accepted/completed/aborted/failed workload |
| `inferx_requests_running`, `inferx_requests_waiting` | gauge | Scheduler pressure |
| `inferx_prompt_tokens_total`, `inferx_generation_tokens_total` | counter | Input/output throughput |
| `inferx_cached_prompt_tokens_total` | counter | Prefix-cache effectiveness |
| `inferx_request_queue_seconds` | histogram | Admission delay |
| `inferx_time_to_first_token_seconds` | histogram | User-visible prefill latency |
| `inferx_inter_token_latency_seconds` | histogram | Decode cadence, including stalls |
| `inferx_request_duration_seconds` | histogram | End-to-end latency |
| `inferx_request_prompt_tokens`, `inferx_request_generation_tokens` | histogram | Workload-shape correlation |
| `inferx_batch_sequences`, `inferx_batch_tokens` | histogram | Batch composition |
| `inferx_steps_total{phase,graph_mode}` | counter | Scheduler/model progress |
| `inferx_engine_step_seconds{phase,rank}` | histogram | Per-rank step cost and imbalance |
| `inferx_preemptions_total` | counter | Capacity/scheduling failure pressure |
| `inferx_http_responses_total{endpoint,outcome}` | counter | API errors independent of model completion |

TTFT is measured from request acceptance to first token becoming available to the response path. Queue time ends when the request is admitted to its first model batch. ITL is the interval between consecutive emitted tokens for one request. Request duration ends on completion, abort, or error. All host timestamps use `std::chrono::steady_clock`; wall time is never used for durations.

### KV and prefix cache

| Metric | Type | Purpose |
|---|---|---|
| `inferx_kv_blocks{state}` | gauge | Total/used/free/cached block counts |
| `inferx_kv_cache_usage_ratio` | gauge | Capacity headroom |
| `inferx_prefix_cache_hits_total`, `inferx_prefix_cache_misses_total` | counter | Derive hit rate over any query window |
| `inferx_prefix_cache_evictions_total` | counter | Cache churn |

Hit ratio should be calculated in PromQL from counter rates; an instantaneous ratio gauge is optional and must not replace the counters.

### Tensor parallel and communicator

| Metric | Type | Purpose |
|---|---|---|
| `inferx_communicator_info{backend,world_size}` | gauge | Active topology/configuration |
| `inferx_rank_healthy{rank}` | gauge | Rank liveness |
| `inferx_rank_last_progress_seconds{rank}` | gauge | Detect stalled ranks |
| `inferx_collectives_total{backend,op,rank}` | counter | Collective call mix |
| `inferx_collective_bytes_total{backend,op,rank}` | counter | Communication volume |
| `inferx_collective_failures_total{backend,op,rank}` | counter | NCCL/API failures |
| `inferx_communicator_aborts_total{backend}` | counter | Fatal communicator resets |
| `inferx_rank_timeouts_total{rank}` | counter | Watchdog failures |
| `inferx_collective_seconds{backend,op,rank}` | histogram | Sampled diagnostic latency only |

The core TP dashboard derives rank skew from per-rank step histograms and progress gauges. This is preferable to a high-cardinality metric for every step.

NCCL launches are asynchronous, so host-call duration is not collective duration. Normal mode records only call/byte/error counters and existing step-level CUDA timing. Diagnostic mode may sample (default 1/128, configurable) paired CUDA events into a fixed per-rank ring buffer; events are resolved after an already-required synchronization or on a collector thread. Instrumentation must never add `cudaDeviceSynchronize`, block a collective, allocate in the hot path, or change CUDA graph capture. Nsight Systems/NCCL tests remain the source of truth for one-off microanalysis.

### CUDA graphs and process lifecycle

Add capture/replay/failure counters by `graph_mode`, active graph count, model-load duration, process start time, and `inferx_engine_info{version,model,precision,tp_backend}`. Info labels change only when the process configuration changes.

### GPU host telemetry

DCGM Exporter supplies GPU utilization, framebuffer memory, SM/memory clocks, temperature, power, throttle reasons, PCIe traffic, NVLink traffic when supported, and XID errors. These signals should not be duplicated inside InferX. Profiling fields such as tensor activity are optional because availability and collection overhead vary by GPU and DCGM configuration.

## Histogram buckets

Defaults intentionally retain fine resolution for local low-latency serving while covering long prompts:

- TTFT seconds: `.001,.002,.005,.01,.02,.04,.06,.08,.1,.2,.4,.8,1,2,4,8,16,32,64`
- ITL seconds: `.001,.002,.004,.006,.008,.01,.015,.02,.025,.03,.04,.06,.08,.1,.2,.4,1,2`
- queue seconds: `.0001,.001,.005,.01,.05,.1,.25,.5,1,2.5,5,10,30,60`
- end-to-end seconds: `.1,.25,.5,1,2.5,5,10,20,40,60,120,300,600`
- step/diagnostic collective seconds: `.00001,.000025,.00005,.0001,.00025,.0005,.001,.0025,.005,.01,.025,.05,.1`
- token/sequence counts: `1,2,4,8,16,32,64,128,256,512,1024,2048,4096,8192,16384,32768`

Buckets are configuration constants, not runtime labels. Benchmark reports must record the bucket version because percentile accuracy depends on these boundaries.

## Collection and exposition

Metric writes use single-writer rank/scheduler slots where ownership already exists, otherwise relaxed atomics. Histogram observations increment a fixed bucket counter, count, and sum without maps, allocation, or formatting. The scrape path copies a consistent-enough snapshot under a short registry lock; it performs no CUDA calls and never acquires scheduler or communicator locks. Formatting happens after the snapshot.

`GET /metrics` returns `200` with `Content-Type: text/plain; version=0.0.4; charset=utf-8`. Unsupported methods return the existing API error convention. Prometheus scrape failures must not affect serving. A later OpenMetrics encoder can be added behind the same registry.

Recommended scrape interval is 5 seconds for controlled benchmarks and 15 seconds for longer-running environments. Start with 24-hour local retention or 7-day cloud retention. Recording rules should precompute expensive dashboard quantiles and rates for longer experiments.

Representative queries:

```promql
rate(inferx_generation_tokens_total[1m])

histogram_quantile(0.99,
  sum by (le) (rate(inferx_time_to_first_token_seconds_bucket[5m])))

sum(rate(inferx_prefix_cache_hits_total[5m]))
/
clamp_min(sum(rate(inferx_prefix_cache_hits_total[5m])
  + rate(inferx_prefix_cache_misses_total[5m])), 1)

max by (instance) (inferx_rank_last_progress_seconds)
- min by (instance) (inferx_rank_last_progress_seconds)
```

## Grafana visualization plan

Provision these dashboards from JSON in source control:

1. **Serving overview/SLO** — request rate and failures, prompt/output tokens per second, p50/p95/p99 TTFT/ITL/end-to-end latency, running/waiting requests, and current model/configuration.
2. **Scheduler and KV cache** — queue latency, batch sequences/tokens, phase step latency, KV occupancy, prefix hit rate, evictions, and preemptions.
3. **Tensor parallel/NCCL** — world size/backend, rank health and progress, per-rank p95 step duration, max/min rank skew, collective calls/bytes/error rate, sampled collective latency, timeouts, and communicator aborts.
4. **GPU hardware** — DCGM utilization, memory, clocks, power, temperature, throttle state, PCIe/NVLink traffic, and XID events aligned with InferX latency.
5. **Benchmark comparison** — variables for instance, model, precision, TP size, backend, and graph mode; throughput/latency/cache/communication panels suitable for before/after comparisons.

Annotations mark process restarts, model loads, benchmark phases, and configuration changes. Dashboard units, rate windows, and “no data” behavior are explicit. Panels link rank-level anomalies to the matching GPU/DCGM series.

Initial alerts: InferX scrape missing, any collective failure/communicator abort/rank timeout, rank progress stalled, DCGM XID error, sustained thermal/power throttling, GPU memory above 95%, growing queue with flat throughput, and sustained p99 TTFT regression. Thresholds for latency and queue alerts are workload-specific and must be calibrated from a known-good run rather than hard-coded globally.

## Planned repository layout

```text
include/inferx/observe/metrics.h
src/observe/metrics.cc
tests/observe/metrics_test.cc
observe/
  prometheus/prometheus.yml
  prometheus/rules.yml
  grafana/provisioning/datasources/prometheus.yml
  grafana/provisioning/dashboards/default.yml
  grafana/dashboards/*.json
  docker-compose.yml
third_party/prometheus/                 # pinned service source
third_party/grafana/                    # pinned service source
```

DCGM Exporter and optional Node Exporter will be pinned as container images in the deployment manifest, not vendored into the C++ dependency graph.

## Implementation milestones

1. **Registry and endpoint** — implement counter/gauge/histogram primitives, escaping and text exposition; add `/metrics`; preserve `/stats`; add golden-format and concurrent-scrape tests.
2. **Serving lifecycle** — attach request timestamps and terminal outcomes; expose scheduler, token, latency, batch, KV, prefix-cache, preemption, graph, and engine-info metrics from their current owners.
3. **Communicator hooks** — add backend-neutral callbacks for call/byte/failure/abort/timeout accounting; instrument NCCL without timing synchronization; add rank health and per-rank step timing.
4. **Diagnostic timing** — implemented behind the communicator interface and
   enabled with `--collective-timing-sample-rate N`. It exports completed
   samples, graph skips, and ring/event drops; zero (the default) performs no
   CUDA timing calls.
5. **Monitoring stack** — implemented as a pinned Compose deployment with
   Prometheus scrape/recording/alert rules, provisioned Grafana dashboards,
   optional DCGM Exporter and Node Exporter profiles, and rank/GPU selectors.
6. **Validation** — test single-GPU baselines first, then one-host/two-GPU TP under prompt-heavy, decode-heavy, saturated, cache-hit, cache-miss, cancellation, and injected rank-failure scenarios.

## Acceptance criteria

- Every metric has stable HELP/TYPE text, valid escaping, bounded labels, and documented units/semantics.
- Histogram buckets are cumulative, include `+Inf`, and have correct `_count`/`_sum`; concurrency tests pass under ThreadSanitizer where available.
- Counter resets are explainable by `process_start_time_seconds`; no request-derived label can increase series cardinality.
- Scraping is non-mutating and performs no CUDA/communicator operations. A failed or slow scraper cannot delay inference.
- Monitoring disabled versus enabled changes steady-state throughput and p99 latency by less than 1% in repeated runs; diagnostic collective sampling has a separately reported overhead budget.
- On two GPUs, dashboards show both ranks/GPUs, correlate rank to GPU UUID, expose a deliberately injected communicator error, and make a stalled or slower rank visible.
- Dashboard queries return meaningful zero/no-data states after process start, during idle periods, and after restart.

## Decisions and open items

- The service repositories are pinned submodules, but production uses tagged images matching those revisions. Their sources are not CMake dependencies.
- Prometheus is the pull-based source of truth. `/stats` is retained but Grafana does not consume it.
- Per-collective GPU timing is opt-in and sampled. Always-on per-rank step timing plus communication counters is the safe default.
- Before deployment, select DCGM Exporter and Node Exporter image versions compatible with the rented host driver and GPU generation.
- Before freezing alerts, establish baselines for the selected model, maximum sequence length, precision, CUDA graph mode, and expected request mix.

## References

- Local vLLM metric definitions: `/home/gc/vllm/vllm/engine/metrics.py`
- Local SGLang worker metrics: `/home/gc/sglang/python/sglang/srt/observability/metrics_collector.py`
- Local SGLang router metrics: `/home/gc/sglang/experimental/sgl-router/src/server/metrics.rs`
- Prometheus: <https://github.com/prometheus/prometheus>
- Grafana: <https://github.com/grafana/grafana>
- NVIDIA DCGM Exporter: <https://github.com/NVIDIA/dcgm-exporter>
