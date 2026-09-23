# Benchmarks

A small benchmark framework: generic infrastructure under
[`inferx_bench/`](inferx_bench/), one directory per model (currently
[`qwen3/`](qwen3/)) holding that model's workload, presets and entry-point
wrappers, and a single CLI ([`bench.py`](bench.py)) that composes the two.
The design follows the separation vLLM and SGLang converge on — a CLI with
explicit experimental parameters, pluggable datasets/workloads, engine
adapters, and results saved as self-describing JSON — but stays
proportionate to this repository: no client-server framework of its own,
one load generator (`vllm bench serve`) reused rather than reimplemented.

## Directory structure

```
benchmarks/
  bench.py              CLI entry point (workload | analyze | serve)
  prompts.json          Shared token-ID prompt corpus (workload source material)
  inferx_bench/         Generic infrastructure — no model-specific code
    models.py           Model registry: one ModelConfig per checkpoint
    workload.py         Workload-suite schema/loading (token IDs, capacity knobs)
    runner.py           Offline driver: GPU lock, NVML monitor, provenance
    vllm_worker.py      vLLM in-process suite worker (incremental timings)
    serve.py            Serving benchmark: server lifecycle, load gen, compare
    metrics.py          Metric computation from one raw trial
    analyze.py          Tag comparison: summaries, agreement, plots
    validate.py         Fail-closed data validation (GPU-free)
    provenance.py       Digests, environment capture, source archives
  qwen3/                Qwen3-0.6B: frozen workload, presets, wrappers, records
    run.py              Offline suite run (qwen3 defaults; wraps runner)
    analyze.py          Tag comparison (wraps inferx_bench.analyze)
    serve_bench.py      Serving comparison under server.md section 9 conditions
    make_workload.py    How workload.json was generated (do not regenerate)
    numerical.py        Logit-replay diagnostics vs Transformers
    results/            Archived offline runs, one directory per tag
    serve_results/      Archived serving runs, one directory per tag
```

Run archives (results directories and their plots/logs) are generated
artifacts and gitignored; the framework, the frozen workload, and the
documentation are source.

## How to run a benchmark

All commands run from the repository root. Anything that touches vLLM or
NVML needs the repo venv: `python/.venv/bin/python`. Never run engines
concurrently; both drivers take an exclusive lock, but unrelated GPU
processes are your responsibility.

Offline suite (fixed token-ID workload through the real engine, warmup +
measured repeats, per-run provenance):

```sh
cmake --build build-cuda13 --target inferx -j 8
python/.venv/bin/python benchmarks/bench.py workload --model qwen3-0.6b \
  --engine inferx --tag my_run
python/.venv/bin/python benchmarks/bench.py workload --model qwen3-0.6b \
  --engine vllm --tag my_vllm_baseline
```

Compare archived tags (tables, agreement, plots land in `--output-dir`):

```sh
MPLCONFIGDIR=/tmp/mpl python/.venv/bin/python benchmarks/bench.py analyze \
  --model qwen3-0.6b my_vllm_baseline my_run --require-exact
```

Serving benchmark (starts a real server, drives it with `vllm bench serve`,
plus a greedy agreement probe):

```sh
python/.venv/bin/python benchmarks/bench.py serve --model qwen3-0.6b \
  --engine inferx --tag serve_smoke --cells 128:32:16:64,512:128:1:16
python/.venv/bin/python benchmarks/bench.py serve --model qwen3-0.6b \
  --compare serve_smoke_vllm serve_smoke
```

Key parameters are always explicit on the CLI: `--model`, `--engine`,
`--tag`, `--suite`, `--repeats`, `--cells` (input_len:output_len:
concurrency:num_prompts), `--inferx-binary`, plus the InferX experimental
engine switches (`--cuda-graphs`, `--decode-linear`, `--split-decode`,
`--packed-projections`, `--prefill-tile`, `--scalar-kv`). Workload cases
carry their own per-case `batch`/`concurrency`/lengths; the C++ `inferx
bench workload/latency/throughput` subcommands remain available for direct
single-shape runs (`--batch-size`, `--input-len`, `--output-len`,
`--num-iters`, `--num-iters-warmup`).

## Benchmarking Qwen3 specifically

The qwen3 directory is the reference model-specific setup: a frozen 24-case
workload (`workload.json`, hash-pinned in every run's metadata), fixed
fairness conditions, and wrappers that keep the historical entry points
working. See [qwen3/README.md](qwen3/README.md) for the full conditions,
the validation commands, and the metric caveats; the entry points are:

```sh
python/.venv/bin/python benchmarks/qwen3/run.py --engine inferx --tag new_tag
python/.venv/bin/python benchmarks/qwen3/analyze.py vllm_baseline inferx_baseline ...
python/.venv/bin/python benchmarks/qwen3/serve_bench.py --engine inferx --tag smoke_tag
```

## Adding another model

1. Register it in [`inferx_bench/models.py`](inferx_bench/models.py): one
   `ModelConfig` with path, dtype, context cap, vocab bound, and KV
   geometry (layers/kv_heads/head_dim) so KV bytes and validation derive
   correctly for both engines.
2. Optionally create `benchmarks/<name>/` with a `workload.json` (token-ID
   cases; `make_workload.py` in qwen3 shows the schema) and thin wrappers
   if you want frozen presets like qwen3's.
3. Run `benchmarks/bench.py workload --model <name> ...`.

No benchmark logic is copied: the runner, worker, validation, metrics and
analysis are engine- and model-generic. Requirements: the checkpoint must
load in both engines (for cross-engine comparison) or at least in InferX
(single-engine runs work fine), and its tokenizer must produce token IDs
compatible with the chosen workload corpus.

## Metrics

Per trial, then median across repeats (definitions are service-level and
deliberately include scheduling and queueing; see qwen3/README.md for the
caveats that still apply):

| Metric | Meaning |
|---|---|
| `ttft_ms` / `ttft_p95_ms` | Time to first token per request (all requests are submitted together, so this includes queueing) |
| `itl_ms` / `itl_p95_ms` | Inter-token latency from consecutive real token arrivals |
| `e2e_ms` | Wall-clock span from submission to the last token |
| `prefill_tokens_s` | Input tokens / time until every request produced its first token |
| `decode_tokens_s` | Output tokens after the first / first-to-last-token window |
| `e2e_tokens_s` | All output tokens / end-to-end span (total throughput) |
| `input_tokens` / `output_tokens` | Exact token counts for the case |
| `batch` / `concurrency` | Scheduled-sequence count and simultaneous requests |
| `peak_device_mib` / `peak_device_delta_mib` | Whole-device NVML peak (and delta over idle baseline), sampled every 5 ms — includes startup; not per-process on WSL |

Serving runs additionally record the `vllm bench serve` metrics
(mean/p99 TTFT/TPOT/ITL, output throughput, success rate) per cell.

## Results and reproducibility

Each run writes `<results-root>/<model>/<tag>/` (qwen3's historical root is
`benchmarks/qwen3/results/`) containing: `metadata.json` (git head, dirty
diff, source archive, checkpoint/config/workload/binary hashes, harness
hashes, GPU/driver/package versions, exact commands, experimental flags),
per-batch logs and memory traces, and the raw trial rows with token
timestamps and outputs. `analyze` refuses to compare tags whose weights,
environment or workload hash differ, and validates every trial's structure
before summarizing; `--require-exact` additionally fails the comparison on
any greedy divergence between engines. Comparative claims should always
name the tags (and hence the recorded conditions) they come from.
