# Qwen3-0.6B batch output comparison — 2026-10-19

Exact output parity failed: **115 / 174 sequences** match; **59 differ** across 24 cases. All outputs have the requested length.

Both engines use the same local checkpoint and raw prompt token IDs, BF16, eager execution, prefix caching enabled, a 4,096-token scheduling budget, 16-token blocks, and 3,758,096,384 KV-cache bytes. Sampling is greedy, with no penalties and EOS ignored to compare fixed output lengths. Each case has one warmup and one measured trial. vLLM is 0.29.0; its log reports FlashAttention 2. InferX uses the current uncommitted ModelRunner/Worker integration.

Batch size 4 includes both 4 and 8 concurrent requests. Prompt lengths are 128, 512, and 1,024; output lengths are 32 and 128. These are the existing frozen workload prompts, truncated/repeated token IDs, not a natural-language quality benchmark.

| Batch capacity | Exact sequences |
|---|---:|
| 1 | 5 / 6 |
| 4 | 44 / 72 |
| 16 | 66 / 96 |

Matching token positions: 11044 / 13920. After the first divergence, subsequent token comparisons are conditioned on different histories; this is not a per-step numerical-error rate.

Example: `p128_o32_b4_c4`, sequence 0, first generated token: InferX token 8 (`)`), vLLM token 594 (`'s`). The cause of divergence has not been diagnosed by this comparison; do not assume BF16 ties explain it.

The benchmark collector initially discarded valid tokens after a warm prefix-cache hit because it counted only newly computed prompt tokens. `src/bench/workload.cc` now consumes the runner’s returned samples directly; the successful runs below use that fix. vLLM initially selected the system CUDA compiler, which rejected a FlashInfer JIT flag. The successful reference runs set `CUDA_HOME=/usr/local/cuda` and put its `bin` first in PATH.

Artifacts: `comparison.json` contains every mismatch, first differing token IDs, and decoded outputs; `*_b*.log` contain raw token arrays. `metadata.json` records commands, versions, hashes, and settings. `source.patch` records source changes. `vllm_worker.py` is the existing worker adapted to eager mode with prefix caching enabled. Recompute the comparison from the repository root with:

```sh
python/.venv/bin/python benchmarks/qwen3/runner_parity_20261019/compare.py
```

The InferX server was restarted on 127.0.0.1:8000 and its health endpoint verified after the runs.

| Case | Exact sequences |
|---|---:|
| p128_o32_b1_c1 | 1 / 1 |
| p128_o32_b4_c4 | 2 / 4 |
| p128_o32_b4_c8 | 5 / 8 |
| p128_o32_b16_c16 | 10 / 16 |
| p128_o128_b1_c1 | 1 / 1 |
| p128_o128_b4_c4 | 2 / 4 |
| p128_o128_b4_c8 | 4 / 8 |
| p128_o128_b16_c16 | 9 / 16 |
| p512_o32_b1_c1 | 1 / 1 |
| p512_o32_b4_c4 | 2 / 4 |
| p512_o32_b4_c8 | 4 / 8 |
| p512_o32_b16_c16 | 14 / 16 |
| p512_o128_b1_c1 | 1 / 1 |
| p512_o128_b4_c4 | 2 / 4 |
| p512_o128_b4_c8 | 4 / 8 |
| p512_o128_b16_c16 | 13 / 16 |
| p1024_o32_b1_c1 | 1 / 1 |
| p1024_o32_b4_c4 | 4 / 4 |
| p1024_o32_b4_c8 | 7 / 8 |
| p1024_o32_b16_c16 | 14 / 16 |
| p1024_o128_b1_c1 | 0 / 1 |
| p1024_o128_b4_c4 | 3 / 4 |
| p1024_o128_b4_c8 | 5 / 8 |
| p1024_o128_b16_c16 | 6 / 16 |
