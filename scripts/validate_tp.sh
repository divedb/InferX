#!/bin/bash
# Reproducible M7 hardware acceptance for a one-host, two-GPU NCCL deployment.
# It deliberately records failed preflight checks in the report before exiting.
set -uo pipefail

cd "$(dirname "$0")/.." || exit 1
ROOT=$(pwd)
BUILD_DIR=${BUILD_DIR:-$ROOT/build-cuda}
RESULT_ROOT=${RESULT_ROOT:-$ROOT/validation-results}
RUN_ID=$(date +%Y%m%d-%H%M%S)
RESULT_DIR=$RESULT_ROOT/tp-$RUN_ID
MODEL_DIR=${MODEL_DIR:-}
MODEL_NAME=${MODEL_NAME:-Qwen2.5-3B-Instruct}
RUN_SERVING=${RUN_SERVING:-0}
SCRAPER_PID=""
mkdir -p "$RESULT_DIR"
REPORT=$RESULT_DIR/report.txt

exec > >(tee -a "$REPORT") 2>&1

section() {
  echo
  echo "## $1"
}

run_recorded() {
  echo "+ $*"
  "$@"
  local status=$?
  echo "exit_status=$status"
  return "$status"
}

stop_scraper() {
  if [[ -n "$SCRAPER_PID" ]]; then
    kill "$SCRAPER_PID" 2>/dev/null || true
    wait "$SCRAPER_PID" 2>/dev/null || true
    SCRAPER_PID=""
  fi
}

start_scraper() {
  (
    while true; do
      curl --silent --show-error --max-time 2 \
        http://127.0.0.1:8000/metrics >/dev/null 2>&1 || true
      sleep 5
    done
  ) &
  SCRAPER_PID=$!
}

trap stop_scraper EXIT INT TERM

section "Run metadata"
echo "timestamp=$(date --iso-8601=seconds)"
echo "git_commit=$(git rev-parse HEAD 2>/dev/null || echo unknown)"
echo "build_dir=$BUILD_DIR"
echo "model_name=$MODEL_NAME"
echo "run_serving=$RUN_SERVING"
run_recorded uname -a || true

section "CUDA and NCCL software"
run_recorded "$BUILD_DIR/src/server/inferx-serve" --help || true
if command -v nvcc >/dev/null 2>&1; then run_recorded nvcc --version || true; fi
if command -v ldconfig >/dev/null 2>&1; then
  run_recorded /bin/bash -c "ldconfig -p 2>/dev/null | grep -E 'libnccl|libcuda'" || true
fi

section "GPU inventory and topology"
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "FAIL: nvidia-smi is not installed"
  echo "result_dir=$RESULT_DIR"
  exit 2
fi
if ! run_recorded nvidia-smi --query-gpu=index,name,uuid,memory.total,driver_version --format=csv,noheader; then
  echo "FAIL: NVIDIA devices are not accessible"
  echo "result_dir=$RESULT_DIR"
  exit 2
fi
GPU_COUNT=$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l)
echo "gpu_count=$GPU_COUNT"
run_recorded nvidia-smi topo -m || true
if [[ "$GPU_COUNT" -lt 2 ]]; then
  echo "FAIL: TP=2 acceptance requires at least two visible GPUs"
  echo "result_dir=$RESULT_DIR"
  exit 2
fi

section "NCCL collective gate"
if [[ ! -x "$BUILD_DIR/tests/communicator_test" ]]; then
  echo "FAIL: $BUILD_DIR/tests/communicator_test does not exist; configure a CUDA/NCCL build first"
  echo "result_dir=$RESULT_DIR"
  exit 2
fi
run_recorded "$BUILD_DIR/tests/communicator_test" \
  --gtest_filter=NcclCommTest.TwoGpuBf16AllReduceUsesTheSuppliedStreams \
  --gtest_output="xml:$RESULT_DIR/nccl-test.xml"
NCCL_STATUS=$?
if [[ "$NCCL_STATUS" -ne 0 ]] || grep -Eq '<skipped|result="skipped"' "$RESULT_DIR/nccl-test.xml" 2>/dev/null; then
  echo "FAIL: the two-GPU NCCL test failed or skipped"
  echo "result_dir=$RESULT_DIR"
  exit 3
fi

section "Host-sim and tensor-sharding regressions"
run_recorded ctest --test-dir "$BUILD_DIR" --output-on-failure \
  -R 'HostSimCommTest|TensorParallelShardTest|CommunicatorMetricsTest' || exit 4

if [[ "$RUN_SERVING" != 1 ]]; then
  section "Serving validation"
  echo "SKIP: set RUN_SERVING=1 and MODEL_DIR=/path/to/checkpoint to run TP=1/TP=2 serving benchmarks"
  echo "PASS: collective and regression gates completed"
  echo "result_dir=$RESULT_DIR"
  exit 0
fi
if [[ -z "$MODEL_DIR" || ! -d "$MODEL_DIR" ]]; then
  echo "FAIL: RUN_SERVING=1 requires an existing MODEL_DIR"
  echo "result_dir=$RESULT_DIR"
  exit 5
fi

section "TP=1 and TP=2 serving matrix"
export MODEL_DIR MODEL_NAME
export ITERS=${ITERS:-5}
export WARMUP=${WARMUP:-1}
export ROUNDS=${ROUNDS:-2}
export CONCURRENCY=${CONCURRENCY:-1,2,4,8}
export PREFILL_LENS=${PREFILL_LENS:-512,2048}
export DECODE_TOKENS=${DECODE_TOKENS:-128}
run_recorded ./scripts/bench_serve.sh inferx-tp1 || exit 6
run_recorded ./scripts/bench_serve.sh inferx-tp2 || exit 7
start_scraper
run_recorded ./scripts/bench_serve.sh inferx-tp2-scraped || exit 8
stop_scraper
run_recorded ./scripts/bench_serve.sh inferx-tp2-sampled || exit 9

section "Result"
echo "PASS: M7 one-host/two-GPU functional and serving validation completed"
echo "Benchmark JSON is under bench-results; report=$REPORT"
