"""Qwen3-0.6B tag comparison; thin wrapper over inferx_bench.analyze.

Commands from README.md and server.md keep working; the summary/agreement/
CSV/plot artifacts and their schema are unchanged, so historical tags and
new tags from the generic runner remain comparable.
"""
import argparse
import sys
from pathlib import Path

QWEN3 = Path(__file__).resolve().parent
sys.path.insert(0, str(QWEN3.parent / "inferx_bench"))

import analyze  # noqa: E402
from models import get_model  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument('tags', nargs='+')
    p.add_argument('--require-exact', action='store_true',
                   help='Exit nonzero after saving artifacts if any greedy output differs')
    p.add_argument('--output-dir', type=Path, default=QWEN3,
                   help='Keep new comparisons separate from historical reports')
    args = p.parse_args()
    analyze.analyze(
        QWEN3 / 'results', QWEN3 / 'workload.json', args.tags,
        vocab_size=get_model('qwen3-0.6b').vocab_size,
        output_dir=args.output_dir, require_exact=args.require_exact,
        plot_title='Qwen3-0.6B · fixed workload · '
                   'trial medians (end-to-end whiskers: min/max)')


if __name__ == '__main__':
    main()
