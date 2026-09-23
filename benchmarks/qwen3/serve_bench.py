"""Qwen3-0.6B serving benchmark; thin wrapper over inferx_bench.serve.

Two-engine `vllm bench serve` comparison under the repo's fixed conditions
(server.md section 9): pinned capacity knobs, greedy + ignore_eos, the
random dataset with a fixed seed, and a greedy agreement probe per engine.
The smoke matrix and the results root are the qwen3-specific parts; server
lifecycle, load generation and comparison live in the generic package.

Usage:
  python/.venv/bin/python benchmarks/qwen3/serve_bench.py --engine inferx --tag smoke_inferx
  python/.venv/bin/python benchmarks/qwen3/serve_bench.py --engine vllm   --tag smoke_vllm
  python/.venv/bin/python benchmarks/qwen3/serve_bench.py --compare smoke_inferx smoke_vllm
"""
import argparse
import fcntl
import sys
from pathlib import Path

QWEN3 = Path(__file__).resolve().parent
sys.path.insert(0, str(QWEN3.parent / "inferx_bench"))

import serve  # noqa: E402
from models import get_model  # noqa: E402

ROOT = QWEN3.parents[1]

# server.md section 9: identical capacity on both engines.
KV_BYTES = 3758096384
MAX_NUM_SEQS = 16
TOKEN_BUDGET = 4096
BLOCK_SIZE = 16

# Smoke matrix: (input_len, output_len, concurrency, num_prompts).
MATRIX = [
    (128, 32, 16, 64),
    (1024, 128, 16, 64),
    (512, 128, 1, 16),
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--engine", choices=["inferx", "vllm"])
    p.add_argument("--tag")
    p.add_argument("--compare", nargs="+")
    args = p.parse_args()
    if args.compare:
        serve.compare(QWEN3 / "serve_results", args.compare)
        return
    if not args.engine or not args.tag:
        p.error("--engine and --tag are required (or --compare tags)")
    lock = open("/tmp/inferx-benchmark.lock", "w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    serve.run_engine(ROOT, get_model("qwen3-0.6b"), args.engine, args.tag, MATRIX,
                     QWEN3 / "serve_results", port=8137,
                     max_num_seqs=MAX_NUM_SEQS, token_budget=TOKEN_BUDGET,
                     block_size=BLOCK_SIZE, kv_bytes=KV_BYTES)


if __name__ == "__main__":
    main()
