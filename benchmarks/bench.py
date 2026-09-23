#!/usr/bin/env python3
"""Benchmark entry point: compose generic infrastructure with a model.

Subcommands:
  workload  Offline suite benchmark through the real engine (GPU, serial,
            one engine at a time, full provenance).
  analyze   Compare archived tags: summaries, agreement, CSV/JSON, plots.
  serve     HTTP serving benchmark via `vllm bench serve` against a real
            server, plus a greedy agreement probe; --compare prints a table.

The model registry (benchmarks/inferx_bench/models.py) is the only source
of model-specific knowledge; benchmarks/<model>/ directories add workloads
and presets on top. Run with the repo's venv python for vLLM/NVML access
(python/.venv/bin/python). All commands run from the repository root.
"""
import argparse
import sys
from pathlib import Path

BENCH_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BENCH_DIR / "inferx_bench"))
sys.path.insert(0, str(BENCH_DIR))

ROOT = BENCH_DIR.parent


def cmd_workload(args):
    import runner
    from models import get_model
    model = get_model(args.model)
    runner.run_suite(
        ROOT, model, args.engine, args.tag, args.suite, args.repeats,
        results_root=ROOT / args.results_root / model.name,
        inferx_binary=args.inferx_binary,
        cuda_graphs=args.cuda_graphs, decode_linear=args.decode_linear,
        split_decode=args.split_decode, packed_projections=args.packed_projections,
        prefill_tile=args.prefill_tile, scalar_kv=args.scalar_kv)


def cmd_analyze(args):
    import analyze
    from models import get_model
    model = get_model(args.model)
    analyze.analyze(ROOT / args.results_root / model.name, args.suite, args.tags,
                    vocab_size=model.vocab_size, output_dir=args.output_dir,
                    require_exact=args.require_exact,
                    plot_title=f"{model.name} · fixed workload · "
                               "trial medians (end-to-end whiskers: min/max)")


def cmd_serve(args):
    import serve
    from models import get_model
    model = get_model(args.model)
    if args.compare:
        serve.compare(ROOT / args.serve_results_root / model.name, args.compare)
        return
    matrix = [tuple(int(v) for v in cell.split(":")) for cell in args.cells.split(",")]
    serve.run_engine(ROOT, model, args.engine, args.tag, matrix,
                     ROOT / args.serve_results_root / model.name,
                     port=args.port)


def add_model_arg(p):
    p.add_argument("--model", required=True,
                   help="Registry key from benchmarks/inferx_bench/models.py")


def main():
    p = argparse.ArgumentParser(prog="bench", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    w = sub.add_parser("workload", help="Offline engine-suite benchmark")
    add_model_arg(w)
    w.add_argument("--engine", choices=["inferx", "vllm"], required=True)
    w.add_argument("--tag", required=True, help="New, unique run tag")
    w.add_argument("--suite", default=None, help="Workload suite JSON")
    w.add_argument("--repeats", type=int, default=3, help="Measured trials per case")
    w.add_argument("--results-root", default="benchmarks/results",
                   help="Run archives land in <root>/<model>/<tag>")
    w.add_argument("--inferx-binary", default="build-cuda13/apps/inferx")
    w.add_argument("--cuda-graphs", action="store_true", help="Experimental")
    w.add_argument("--decode-linear", action="store_true", help="Experimental")
    w.add_argument("--split-decode", action="store_true", help="Experimental")
    w.add_argument("--packed-projections", action="store_true", help="Experimental")
    w.add_argument("--prefill-tile", type=int, choices=(64, 128), default=64)
    w.add_argument("--scalar-kv", action="store_true", help="Diagnostic comparison")
    w.set_defaults(func=cmd_workload)

    a = sub.add_parser("analyze", help="Compare archived tags")
    add_model_arg(a)
    a.add_argument("tags", nargs="+")
    a.add_argument("--suite", default=None)
    a.add_argument("--results-root", default="benchmarks/results")
    a.add_argument("--require-exact", action="store_true",
                   help="Exit nonzero if any greedy output differs")
    a.add_argument("--output-dir", type=Path, default=None)
    a.set_defaults(func=cmd_analyze)

    s = sub.add_parser("serve", help="HTTP serving benchmark")
    add_model_arg(s)
    s.add_argument("--engine", choices=["inferx", "vllm"])
    s.add_argument("--tag")
    s.add_argument("--cells", help="input_len:output_len:concurrency:num_prompts,...")
    s.add_argument("--serve-results-root", default="benchmarks/serve_results")
    s.add_argument("--port", type=int, default=8137)
    s.add_argument("--compare", nargs="+")
    s.set_defaults(func=cmd_serve)

    args = p.parse_args()
    # Default the suite to the model directory's workload when the
    # subcommand uses one.
    if args.command in ("workload", "analyze") and args.suite is None:
        candidate = BENCH_DIR / args.model / "workload.json"
        if not candidate.exists():
            p.error(f"no --suite given and no default workload at {candidate}")
        args.suite = str(candidate)
    args.func(args)


if __name__ == "__main__":
    main()
