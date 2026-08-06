# Tensor-parallel hardware validation

M7 implementation is complete, but acceptance requires a physical machine with
two visible NVIDIA GPUs and an NCCL-enabled build. Run this immediately after
provisioning a candidate host:

```bash
BUILD_DIR=$PWD/build-cuda ./scripts/validate_tp.sh
```

The script records the Git revision, CUDA/NCCL software, GPU UUIDs, memory,
driver, and `nvidia-smi topo -m`; then it runs the hardware-gated BF16 NCCL
all-reduce plus the HostSim, sharding, and communication-metrics regressions.
It fails if the NCCL test skips, so a green report cannot accidentally mean
that no GPU was exercised. Reports are written under `validation-results/`,
which is intentionally ignored by Git.

To include end-to-end serving and scaling measurements:

```bash
RUN_SERVING=1 \
MODEL_DIR=/path/to/Qwen2.5-3B-Instruct \
MODEL_NAME=Qwen2.5-3B-Instruct \
./scripts/validate_tp.sh
```

This runs the same client workload against TP=1, TP=2, TP=2 with a five-second
`/metrics` scrape, and TP=2 with sampled CUDA-event collective timing. The
scraped and unsampled TP=2 rows measure metrics exposition overhead without
conflating it with diagnostic CUDA timing. The matrix covers batch-one latency,
concurrency-driven saturation, 512/2048-token prefill, and decode-heavy generation. Override
`CONCURRENCY`, `PREFILL_LENS`, or `DECODE_TOKENS` to match the rented GPUs'
memory envelope.

## Acceptance review

Do not claim TP scaling from a single number. Retain the raw benchmark JSON and
review all of the following:

- TP=1 and TP=2 produce correct, deterministic completions.
- Both ranks stay healthy and advance; collective failures, aborts, and
  timeouts remain zero in the normal run.
- TP=2 throughput and latency are compared with TP=1 for both prompt-heavy and
  decode-heavy traffic, with topology included alongside the result.
- Scraped and sampled TP=2 runs are each compared with unsampled TP=2.
  Monitoring and diagnostic timing overhead are reported separately; repeat
  runs before enforcing the design target of less than 1% steady-state impact.
- Prometheus sees both ranks and DCGM sees both GPU UUIDs. Rank-to-GPU mapping
  comes from the configured device order (`--devices 0,1`) and the inventory
  captured in the report.
- Cancellation is exercised through the existing server regression. Before
  production, kill or suspend one rank in a controlled test and confirm the
  peer aborts within the 30-second watchdog rather than hanging indefinitely.

The final failure-injection item is intentionally manual today: rank workers
are threads in one process, and adding a production CLI switch that corrupts a
collective would itself be unsafe. A deterministic test-only injection hook is
the next code change if automated chaos coverage is required.
