#!/bin/bash
# Launch vLLM's OpenAI server on Qwen2.5-3B-Instruct for the head-to-head
# benchmark. Logs to /tmp/vllm_server.log. Intended to be run detached, which
# scripts/bench_serve.sh does; run it by hand to inspect startup.
#
# The flags mirror inferx-serve's so the two engines answer the same question:
# bf16 weights, a bf16 KV cache, chunked prefill and prefix caching on in both,
# and the same concurrency cap. vLLM sizes its KV cache from a memory fraction
# rather than a block count; 0.85 leaves it far more KV than inferx's 4096
# blocks, a handicap we accept rather than tune away -- neither engine's cache
# is the binding constraint at these lengths.
#
# VLLM_WSL2_ENABLE_PIN_MEMORY: vLLM disables pinned host memory under WSL2, and
# its 0.26 model runner allocates UVA buffers that require it -- without this
# the engine core dies at startup with "UVA is not available". Pinned memory
# does work on this box's driver, so the opt-in is the fix rather than falling
# back to the older model runner (which would not be vLLM's fast path).
#
# VLLM_USE_FLASHINFER_SAMPLER: FlashInfer's top-k/top-p sampler is JIT-compiled
# at startup and does not build against CUDA 13's CCCL -- `sampling.cuh` calls
# `cub::BlockAdjacentDifference::FlagHeads`, removed in CCCL 3.x. vLLM's own
# PyTorch sampler is the fallback, and the benchmark runs greedy (temperature 0)
# where neither sampler does anything but an argmax.
set -euo pipefail

MODEL="${MODEL:-$HOME/.cache/huggingface/hub/models--Qwen--Qwen2.5-3B-Instruct/snapshots/aa8e72537993ba99e69dfaafa59ed015b17504d1}"
PORT="${PORT:-8001}"
MAX_SEQS="${MAX_SEQS:-8}"

source "$HOME/inferx/.venv-vllm/bin/activate"
export VLLM_WSL2_ENABLE_PIN_MEMORY=1
export VLLM_USE_FLASHINFER_SAMPLER=0

exec vllm serve "$MODEL" \
  --host 127.0.0.1 --port "$PORT" \
  --served-model-name Qwen2.5-3B-Instruct \
  --dtype bfloat16 \
  --max-num-seqs "$MAX_SEQS" \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.85 \
  > /tmp/vllm_server.log 2>&1
