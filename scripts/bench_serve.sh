#!/bin/bash
# M10: run the same serving benchmark against inferx-serve and vLLM, one engine
# at a time, and print the rows next to each other.
#
#   ./scripts/bench_serve.sh                   # every engine, default sweep
#   ./scripts/bench_serve.sh inferx-bf16 vllm  # a subset, in this order
#
# One at a time is not a limitation of the script, it is the measurement: this
# box has one 16 GB GPU and both engines size their KV cache to fill it, so two
# live servers would be measuring each other's memory pressure. Each engine is
# started, waited for, measured, and killed before the next one starts.
#
# Results land in bench-results/<timestamp>/ as one JSON per engine plus the
# console log, so a run can be re-read without re-running it.
set -uo pipefail

cd "$(dirname "$0")/.."
ROOT=$(pwd)

MODEL_DIR="${MODEL_DIR:-$HOME/.cache/huggingface/hub/models--Qwen--Qwen2.5-3B-Instruct/snapshots/aa8e72537993ba99e69dfaafa59ed015b17504d1}"
MODEL_NAME="${MODEL_NAME:-Qwen2.5-3B-Instruct}"
SERVE=${SERVE:-$ROOT/build-cuda/src/server/inferx-serve}
BENCH=$ROOT/bench/serve_bench.py

# Matched settings. Both engines get the same concurrency cap and the same
# context limit; the sweep and the decode length are the same numbers passed to
# the same client.
MAX_SEQS="${MAX_SEQS:-8}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-4096}"
DECODE_TOKENS="${DECODE_TOKENS:-128}"
CONCURRENCY="${CONCURRENCY:-1,2,4,8}"
PREFILL_LENS="${PREFILL_LENS:-512,2048}"
ROUNDS="${ROUNDS:-4}"
ITERS="${ITERS:-10}"
WARMUP="${WARMUP:-2}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
INFERX_SERVER_ARGS="${INFERX_SERVER_ARGS:-}"

ENGINES=("$@")
if [[ ${#ENGINES[@]} -eq 0 ]]; then
  ENGINES=(inferx-bf16 inferx-fp8 vllm)
fi

OUT_DIR="$ROOT/bench-results/$(date +%Y%m%d-%H%M%S)"
mkdir -p "$OUT_DIR"
LOG="$OUT_DIR/console.log"

SERVER_PID=""

# Kill the server and wait for the GPU memory to actually come back. Without the
# wait the next engine starts sizing its KV cache while the last one's
# allocation is still being torn down, and picks a smaller cache for no reason
# the numbers would explain.
stop_server() {
  [[ -z "$SERVER_PID" ]] && return 0
  kill -TERM "-$SERVER_PID" 2>/dev/null || kill -TERM "$SERVER_PID" 2>/dev/null
  for _ in $(seq 1 30); do
    kill -0 "$SERVER_PID" 2>/dev/null || break
    sleep 1
  done
  kill -KILL "-$SERVER_PID" 2>/dev/null
  SERVER_PID=""
  pkill -f "inferx-serve --model" 2>/dev/null
  pkill -f "vllm serve" 2>/dev/null
  sleep 8
}

trap 'stop_server' EXIT INT TERM

start_inferx() {  # $1: extra inferx-serve flags
  setsid "$SERVE" --model "$MODEL_DIR" --port 8000 \
    --served-model-name "$MODEL_NAME" \
    --max-running "$MAX_SEQS" --max-seq-len "$MAX_SEQ_LEN" \
    $1 $INFERX_SERVER_ARGS > "$OUT_DIR/inferx_server.log" 2>&1 &
  SERVER_PID=$!
}

start_vllm() {
  MODEL="$MODEL_DIR" MODEL_NAME="$MODEL_NAME" MAX_SEQS="$MAX_SEQS" \
    MAX_SEQ_LEN="$MAX_SEQ_LEN" \
    GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}" PORT=8001 \
    VLLM_EXTRA_ARGS="${VLLM_EXTRA_ARGS:-}" \
    setsid bash "$ROOT/scripts/run_vllm.sh" &
  SERVER_PID=$!
}

run_engine() {  # $1: label, $2: endpoint
  python3 "$BENCH" \
    --endpoint "$2" --model "$MODEL_NAME" --label "$1" \
    --decode-tokens "$DECODE_TOKENS" --concurrency "$CONCURRENCY" \
    --prefill-lens "$PREFILL_LENS" --rounds "$ROUNDS" \
    --iters "$ITERS" --warmup "$WARMUP" --wait 420 \
    --json "$OUT_DIR/$1.json" $EXTRA_ARGS 2>&1 | tee -a "$LOG"
}

echo "results -> $OUT_DIR" | tee "$LOG"
nvidia-smi --query-gpu=name,memory.total,clocks.max.sm --format=csv,noheader | tee -a "$LOG"
echo "model $MODEL_NAME | max_seqs $MAX_SEQS | decode $DECODE_TOKENS tok |" \
     "concurrency $CONCURRENCY | prefill $PREFILL_LENS" | tee -a "$LOG"
echo | tee -a "$LOG"

for engine in "${ENGINES[@]}"; do
  case "$engine" in
    inferx-bf16) start_inferx "" ;;
    # Weights-only fp8 exists as its own row to attribute a difference to the
    # weights or to the KV cache; --fp8-kv also forces the prefill dequant
    # workaround (§14, M8), so the two are not one knob.
    inferx-fp8w) start_inferx "--fp8" ;;
    inferx-fp8)  start_inferx "--fp8 --fp8-kv" ;;
    vllm)        start_vllm ;;
    *) echo "unknown engine: $engine" >&2; exit 2 ;;
  esac

  case "$engine" in
    vllm) endpoint=http://127.0.0.1:8001 ;;
    *)    endpoint=http://127.0.0.1:8000 ;;
  esac

  run_engine "$engine" "$endpoint"
  stop_server
  echo | tee -a "$LOG"
done

echo "results in $OUT_DIR" | tee -a "$LOG"
