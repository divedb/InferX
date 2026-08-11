#!/bin/bash
# Launch vLLM's OpenAI server for the head-to-head benchmark. The defaults use
# Qwen2.5-3B-Instruct; scripts/bench_serve.sh passes MODEL and MODEL_NAME so the
# same launcher can also close M11's gpt-oss-20b benchmark exit criterion.
# Logs to /tmp/vllm_server.log. Intended to be run detached, which the benchmark
# driver does; run it by hand to inspect startup.
#
# The flags are the same as inferx-serve's (inferx-serve follows vLLM's flag
# names) so the two engines answer the same question: bf16 weights, a bf16 KV
# cache, chunked prefill and prefix caching on in both, and the same
# concurrency cap. Both engines now size their KV cache from a memory fraction
# (--gpu-memory-utilization; inferx-serve defaults to 0.92, this launcher's
# 0.85 for vLLM) -- neither engine's cache is the binding constraint at these
# lengths. Pin inferx's cache with --num-gpu-blocks-override if a fixed block
# count is wanted.
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

ROOT=$(cd "$(dirname "$0")/.." && pwd)

MODEL="${MODEL:-$HOME/.cache/huggingface/hub/models--Qwen--Qwen2.5-3B-Instruct/snapshots/aa8e72537993ba99e69dfaafa59ed015b17504d1}"
MODEL_NAME="${MODEL_NAME:-Qwen2.5-3B-Instruct}"
PORT="${PORT:-8001}"
MAX_SEQS="${MAX_SEQS:-8}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-4096}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
VLLM_EXTRA_ARGS="${VLLM_EXTRA_ARGS:-}"

source "$ROOT/.venv-vllm/bin/activate"
export VLLM_WSL2_ENABLE_PIN_MEMORY=1
export VLLM_USE_FLASHINFER_SAMPLER=0

exec "$ROOT/.venv-vllm/bin/python" "$ROOT/.venv-vllm/bin/vllm" serve "$MODEL" \
  --host 127.0.0.1 --port "$PORT" \
  --served-model-name "$MODEL_NAME" \
  --dtype bfloat16 \
  --max-num-seqs "$MAX_SEQS" \
  --max-model-len "$MAX_SEQ_LEN" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  $VLLM_EXTRA_ARGS \
  > /tmp/vllm_server.log 2>&1
