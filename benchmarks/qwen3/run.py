"""Qwen3-0.6B offline suite driver; thin wrapper over inferx_bench.runner.

Keeps the historical entry point (commands in README.md and the archived
run metadata reference it) while the implementation lives in the generic
package. Defaults pin the frozen qwen3 workload and the qwen3 results root
so archived tags and new tags remain comparable in place.
"""
import argparse
import sys
from pathlib import Path

QWEN3 = Path(__file__).resolve().parent
sys.path.insert(0, str(QWEN3.parent / "inferx_bench"))

import runner  # noqa: E402
from models import get_model  # noqa: E402

ROOT = QWEN3.parents[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--engine', choices=['vllm', 'inferx'], required=True)
    p.add_argument('--tag', required=True)
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--suite', default='benchmarks/qwen3/workload.json')
    p.add_argument('--flash-attention', action='store_true',
                   help='Deprecated compatibility flag; FlashInfer is now the default')
    p.add_argument('--inferx-binary', default='build-cuda13/apps/inferx')
    p.add_argument('--cuda-graphs', action='store_true',
                   help='Experimental; full matrix validation still pending')
    p.add_argument('--decode-linear', action='store_true',
                   help='Experimental small-batch BF16 projections')
    p.add_argument('--split-decode', action='store_true',
                   help='Experimental partitioned decode attention')
    p.add_argument('--packed-projections', action='store_true',
                   help='Experimental merged QKV and gate/up projections')
    p.add_argument('--prefill-tile', type=int, choices=(64, 128), default=64)
    p.add_argument('--scalar-kv', action='store_true',
                   help='Diagnostic comparison with scalar KV writes')
    args = p.parse_args()
    if args.repeats < 1:
        p.error('--repeats must be positive')
    if args.engine == 'vllm' and (args.flash_attention or args.cuda_graphs
                                  or args.decode_linear or args.split_decode
                                  or args.packed_projections
                                  or args.prefill_tile != 64 or args.scalar_kv):
        p.error('InferX experimental flags do not apply to vLLM')
    runner.run_suite(
        ROOT, get_model('qwen3-0.6b'), args.engine, args.tag,
        args.suite, args.repeats,
        results_root=QWEN3 / 'results',
        inferx_binary=args.inferx_binary,
        cuda_graphs=args.cuda_graphs, decode_linear=args.decode_linear,
        split_decode=args.split_decode, packed_projections=args.packed_projections,
        prefill_tile=args.prefill_tile, scalar_kv=args.scalar_kv,
        worker_module='benchmarks/inferx_bench/vllm_worker.py',
        harness_dirs=[QWEN3])


if __name__ == '__main__':
    main()
