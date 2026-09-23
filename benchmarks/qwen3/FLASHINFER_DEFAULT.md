# FlashInfer default attention — 2026-09-22

CUDA dense-model attention now defaults to FlashInfer. The scalar
`PagedAttentionKernel` and its operation entry points have been removed. Paged KV
cache writes remain; they are separate from attention computation.

`--attention-backend flashinfer` is passed through the runner, model registry,
family loader and decoder configuration. `default` and `flash` are compatibility
aliases for the same backend. Unknown names are rejected instead of silently
ignored. The old `INFERX_EXPERIMENTAL_FLASH_ATTENTION` variable no longer controls
execution; the benchmark and numerical drivers record the actual default.

## Supported integration

- BF16 queries, keys, values and output; full causal attention.
- Head dimensions 64, 128 and 256; query/KV-head ratios 1 through 32.
- Full prefill, cached-prefix chunks, mixed prefill/decode batches and decode.
- Ragged sequences, shuffled physical pages, partial last pages and CUDA graphs.
- The specialized FlashInfer decode kernel handles ratios 1/2/3/4/6/8. Other
  supported ratios use FlashInfer's causal prefill kernel with one query per
  sequence. They do not fall back to the removed kernel.
- Existing optional split-decode and 128-row prefill experiments remain opt-in.

Unsupported head dimensions, ratios above 32 and sliding-window attention return
an explicit status. Model loading validates geometry before uploading weights.
These are limits of this integration, not claims about all FlashInfer backends.
The model builders already reject sliding-window and unsupported recurrent model
execution. Device metadata values must satisfy the documented ragged-batch
contract; public operator entry points validate ranks, shapes, dtypes, devices,
workspace capacities and scalar parameters before launching kernels.

No supported workload tested here required CUTLASS/CuTe. A future fallback should
be added for a concrete required pattern that FlashInfer cannot support or cannot
meet requirements for; no placeholder backend or scalar fallback is advertised.
Small synthetic head dimensions formerly handled by the scalar kernel are no
longer supported by the CUDA attention operation.

## Correctness evidence

The [validation artifacts](flashinfer_default_20260922/) record commands, binary
identity, raw outputs and logs. Production model binary SHA256 remained unchanged
through all measurements.

1. All eight relevant CTest targets passed: attention, operators, model loading /
   forward, model runner, scheduler, scheduler output, CLI arguments and CLI output.
   The 13 existing Python numerical/archive validation tests also passed.
2. The expanded [attention tests](../../tests/attention_test.cc) compare every
   output element against an independent FP64 causal-softmax calculation over
   BF16-rounded inputs, using the existing **0.005 absolute tolerance**, unchanged.
   Coverage includes 128 attention configurations: MHA, MQA, GQA ratios 1..32,
   head sizes 64/128/256, pages 16/32/64, full and partial pages, chunked and mixed
   batches, 1024-token prefill, contexts up to 32768, 32-sequence decode, large
   score scaling, and graph replay after changing page tables / sequence lengths.
   Existing operator tests additionally cover 128-row tiles and cache writes.
3. Compute Sanitizer memcheck passed the expanded attention suite with **zero
   errors and zero leaked allocations**. See
   [memcheck_extended.log](flashinfer_default_20260922/memcheck_extended.log).
   This does not validate the server's CPU allocation/lifetime behavior.
4. The full Qwen3-0.6B fixed workload passed with both eager and CUDA graph
   execution: **24 configurations × 3 repetitions per mode**, each preceded by
   a full warmup. Lengths were 128/512/1024 input, 32/128 output, with scheduling /
   concurrency pairs (1,1), (4,4), (4,8), (16,16). **522/522 generated sequences
   matched exactly between modes**; each mode was deterministic over repetitions.
   This exercises chunked prefill, queueing, request completion and page turnover.
   [Exact agreement](flashinfer_default_20260922/comparison/agreement.json),
   [metrics](flashinfer_default_20260922/comparison/summary.csv).
5. An independent Transformers BF16 SDPA comparison used identical prefixes at
   lengths 128/512/1024, each with zero or 17 teacher-forced continuation tokens.
   All six predictions matched the reference with both full and 16-token chunked
   prefill (**12/12 distinct prefix/chunk comparisons**). Identity, reversed and
   shuffled KV-page mappings produced **bitwise-identical InferX logits**.
   [Reference report](flashinfer_default_20260922/reference/comparison.json).

The SDPA comparison is diagnostic, not a full-model numerical tolerance pass:
logits are not bitwise equal. Maximum absolute logit error was 0.5 for full
prefill and 0.46875 for 16-token chunks; per-prefix RMS errors are retained in the
report. No logit tolerance was widened or inferred from the attention tolerance.
These sampled greedy matches do not establish universal equivalence with SDPA or
vLLM. The only real checkpoint evaluated end-to-end here is Qwen3-0.6B; other
supported dense-model geometries have operator-level coverage.

## Performance and HTTP status

The first fresh `vllm bench serve` run against the new default completed all
**144/144 requests without failures**, with expected input/output token totals:

| Input / output / concurrency | Output tokens/s | Mean TTFT (ms) |
|---|---:|---:|
| 128 / 32 / 16 | 2082.89 | 100.21 |
| 1024 / 128 / 16 | 1649.24 | 213.32 |
| 512 / 128 / 1 | 246.22 | 20.80 |

The 1024/128 case is about **4.09×** the earlier scalar-default result (403.09
output tokens/s), and consistent with the earlier flash-only experiment.
This is a single HTTP trial, not a statistical performance claim. CUDA graphs
were disabled in both HTTP comparisons. Full offline trial ranges are in the
linked metrics. [HTTP logs and JSON](serve_results/flashinfer_default_20260922/).

The environment's `127.*` proxy bypass did not cover the client's localhost
request during startup; a pending proxy connection delayed the initial run before
measurement. The driver now explicitly bypasses proxies for `127.0.0.1` and
`localhost`. A subsequent direct-connection confirmation **stalled at 63/64
completed requests in its first case** and was interrupted after roughly 90
seconds of measured progress. It produced no completed result JSON and is not
counted as a throughput result. Its server showed socket timeouts; the earlier
heap-corruption report remains a separate unresolved issue. No server source was
modified for the attention replacement.
[Interrupted direct run](serve_results/flashinfer_default_direct_20260922/).

## Reproduction

Run GPU tasks serially; use new output tags. Existing result directories are not
overwritten by the benchmark drivers.

```sh
cmake --build build-cuda13 --target inferx inferx_attention_test inferx_ops_test \
  inferx_model_test inferx_model_runner_test inferx_cli_args_test -j 8
env LD_LIBRARY_PATH=/usr/lib/wsl/lib ctest --test-dir build-cuda13 --output-on-failure \
  -R '^(attention_test|ops_test|model_test|model_runner_test|scheduler_test|scheduler_output_test|cli_args_test|cli_output_test)$'
env LD_LIBRARY_PATH=/usr/lib/wsl/lib /usr/local/cuda/bin/compute-sanitizer \
  --tool memcheck --error-exitcode 1 --leak-check full build-cuda13/tests/inferx_attention_test
python/.venv/bin/python benchmarks/qwen3/run.py --engine inferx --tag NEW_eager
python/.venv/bin/python benchmarks/qwen3/run.py --engine inferx --tag NEW_graphs --cuda-graphs
MPLCONFIGDIR=/tmp/inferx-matplotlib python/.venv/bin/python benchmarks/qwen3/analyze.py \
  NEW_eager NEW_graphs --require-exact --output-dir /tmp/NEW_comparison
python/.venv/bin/python benchmarks/qwen3/serve_bench.py --engine inferx --tag NEW_http
```

The [reference driver](flashinfer_default_20260922/reference_check.py) preserves
teacher-forced fixtures, full-vocabulary FP32 dumps, exact commands and page-order
checks. Change its output directory before reproducing. The comparison's zero
logit tolerance is used to report exact differences; it is not an acceptance
threshold and its `numerical_pass` fields intentionally remain false.
