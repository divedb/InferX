# Qwen3 tensor-parallel serving

The server can shard a dense Qwen3 checkpoint across distinct CUDA GPUs in one
process. One scheduler supplies the same batch, KV block assignments, and request
lifecycle to every rank. NCCL reduces row-parallel projections and gathers logits;
rank zero samples tokens and shares those exact ids with every rank for decoding.

```bash
build-cuda13/apps/inferx serve \
  --model models/Qwen3-0.6B \
  --tensor-parallel-size 2 \
  --device-ids 0,1 \
  --max-num-seqs 4 \
  --num-kv-blocks 256
```

Install an NCCL runtime exposing `libnccl.so.2` on the dynamic loader's search path.
Both requested GPUs must be visible to CUDA. Device ordinals follow
`CUDA_VISIBLE_DEVICES`; omitting `--device-ids` selects ordinals `0..TP-1`.
Explicit device ids must be distinct and their count must equal the TP size.
Each GPU needs memory for its weight shard, local KV cache, and activation workspace.

This initial serving path supports dense `Qwen3ForCausalLM` checkpoints with the
existing BF16 execution path. It requires eager execution: `--cuda-graphs` with
TP greater than one returns an explicit error. It does not distribute execution
across machines or worker processes. `bench workload` uses the same TP runner.

For Qwen3-0.6B at TP=2, each rank holds 8 query heads, 4 KV heads, an intermediate
width of 1536, and 75,968 vocabulary rows. Normalizations and residual activations
remain replicated. The 128-dimensional Q/K norms remain intact per head.

Run the regression tests with:

```bash
ctest --test-dir build-cuda13 \
  -R '^(tensor_parallel_test|model_runner_test|comm_test)$' --output-on-failure
```

The Qwen3 test compares chunked prefill, mixed batches, page transitions, and
multi-step decode against TP=1. Its loopback case executes two rank-local models
on one GPU; it checks identical logits across ranks, matching generated tokens,
and relative logit L2 error below 5%. BF16 partial projections introduce rounding
differences from TP=1. The actual NCCL two-GPU case is skipped when fewer than two
GPUs or NCCL are available; loopback does not validate multi-GPU transport or speed.

The runner aborts the rank group after a failed step. Reconstruct it before
retrying requests; partially updated KV state is not reused.
