"""Replay first-divergence prefixes; fail closed on logits AND greedy IDs.

This is a single-sequence eager diagnostic, not a performance measurement or
a substitute for the batched scheduler/graph matrix. Saved vLLM trials contain
token IDs, not logits; the independent logit oracle here is Transformers.
"""
import argparse
import fcntl
import hashlib
import importlib.metadata
import json
import math
import os
import sys
from pathlib import Path
import subprocess
import tarfile

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'inferx_bench'))
from validate import validate_metadata, validate_trials

ROOT = Path(__file__).resolve().parents[2]


def digest(path):
    with open(path, "rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def prepare(args):
    workload = json.loads(args.workload.read_text())
    ref_meta = json.loads((args.reference / "metadata.json").read_text())
    trials = []
    for directory in (args.reference, args.candidate):
        meta = json.loads((directory / "metadata.json").read_text())
        validate_metadata(meta, ref_meta, args.workload.read_bytes())
        rows = [row for batch in (1, 4, 16)
                for row in json.loads((directory / f"b{batch}.json").read_text())]
        validate_trials(rows, workload["cases"], meta["repeats"])
        trials.append({(row["case"], row["repeat"]): row for row in rows})
    cases = []
    for case in workload["cases"]:
        if args.case and case["id"] != args.case:
            continue
        for repeat in range(ref_meta["repeats"]):
            reference = trials[0][case["id"], repeat]["outputs"]
            candidate = trials[1][case["id"], repeat]["outputs"]
            for request, (prompt, ref, got) in enumerate(zip(case["prompts"], reference, candidate)):
                divergence = next((i for i, (a, b) in enumerate(zip(ref, got)) if a != b), None)
                # Matching sequences still contribute their final prediction;
                # mismatches are examined before generated prefixes can differ.
                step = len(ref) - 1 if divergence is None else divergence
                cases.append({"id": f"{case['id']}/r{repeat}/s{request}",
                              "tokens": prompt + ref[:step], "prompt_length": len(prompt),
                              "step": step, "first_divergence": divergence,
                              "reference_token": ref[step], "candidate_token": got[step]})
    if not cases:
        raise ValueError("No replay cases selected")
    fixture = {"cases": cases, "workload_sha256": digest(args.workload),
               "weights": ref_meta["weights"], "config_sha256": ref_meta["config_sha256"],
               "reference": str(args.reference), "candidate": str(args.candidate)}
    with args.output.open("x") as out:
        json.dump(fixture, out, indent=2)
    print(f"Saved {len(cases)} identical-prefix comparisons to {args.output}")


def compare_logits(reference, candidate, atol, rtol):
    if not all(math.isfinite(t) and t >= 0 for t in (atol, rtol)):
        raise ValueError("Tolerances must be finite and nonnegative")
    reference, candidate = np.asarray(reference), np.asarray(candidate)
    if reference.shape != candidate.shape or reference.ndim != 2 or min(reference.shape) < 1:
        raise ValueError("Expected matching, nonempty [rows, vocabulary] logits")
    if reference.shape[1] < 2 or not (np.isfinite(reference).all() and np.isfinite(candidate).all()):
        raise ValueError("Nonfinite logits or invalid vocabulary")
    reports = []
    for ref, got in zip(reference, candidate):
        ref, got = ref.astype(np.float64), got.astype(np.float64)
        error = np.abs(got - ref)
        failures = error > atol + rtol * np.abs(ref)
        ref_id, got_id = int(ref.argmax()), int(got.argmax())
        top = np.partition(ref, -2)[-2:]
        got_top = np.partition(got, -2)[-2:]
        reports.append({"max_abs_error": float(error.max()),
                        "rms_error": float(np.sqrt(np.mean(error * error))),
                        "out_of_tolerance": int(failures.sum()),
                        "worst_token": int(error.argmax()),
                        "reference_argmax": ref_id, "candidate_argmax": got_id,
                        "reference_margin": float(top.max() - top.min()),
                        "candidate_margin": float(got_top.max() - got_top.min()),
                        "reference_gap_at_candidate": float(ref[ref_id] - ref[got_id]),
                        "numerical_pass": not bool(failures.any()),
                        "exact_argmax": ref_id == got_id})
    return reports


def reference_logits(model_path, fixture, chunk_size, device, output):
    import torch
    from transformers import AutoModelForCausalLM

    # Explicit device selection: a CPU run must never be presented as a GPU reference.
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA reference unavailable; GPU access is required")
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.bfloat16, attn_implementation="sdpa", local_files_only=True
    ).to(device).eval()
    with torch.inference_mode(), output.open("xb") as target:
        for case in fixture["cases"]:
            tokens = case["tokens"]
            start, cache = 0, None
            while start < len(tokens):
                count = min(chunk_size, case["prompt_length"] - start) if start < case["prompt_length"] else 1
                result = model(torch.tensor([tokens[start:start + count]], device=device),
                               past_key_values=cache, use_cache=True, logits_to_keep=1)
                cache = result.past_key_values
                start += count
            result.logits[0, -1].float().cpu().numpy().astype("<f4").tofile(target)


def run(args):
    compare_logits([[0., 1.]], [[0., 1.]], args.atol, args.rtol)  # Validate tolerances first.
    fixture = json.loads(args.fixture.read_text())
    config = json.loads((args.model / "config.json").read_text())
    vocab = config["vocab_size"]
    cases = fixture["cases"]
    if not cases or len({c["id"] for c in cases}) != len(cases):
        raise ValueError("Empty or duplicate replay cases")
    for case in cases:
        tokens = case["tokens"]
        if not (0 < case["prompt_length"] <= len(tokens) <= config["max_position_embeddings"]):
            raise ValueError("Invalid prefix length")
        if not all(type(t) is int and 0 <= t < vocab for t in tokens):
            raise ValueError("Invalid token ID")
    weights = {p.name: digest(p) for p in sorted(args.model.glob("*.safetensors"))}
    if not weights or weights != {Path(k).name: v for k, v in fixture["weights"].items()}:
        raise ValueError("Checkpoint differs from archived reference")
    if digest(args.model / "config.json") != fixture["config_sha256"]:
        raise ValueError("Configuration differs from archived reference")
    # Share the throughput driver's lock; reference and candidate run serially.
    with open("/tmp/inferx-qwen3-benchmark.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        args.output.mkdir(parents=True, exist_ok=False)
        (args.output / "fixture.json").write_bytes(args.fixture.read_bytes())
        metadata = {"fixture_sha256": digest(args.fixture), "weights": weights,
                    "config_sha256": fixture["config_sha256"], "atol": args.atol, "rtol": args.rtol,
                    "reference_device": args.device, "reference_attention": "sdpa",
                    "attention_backend": "flashinfer", "chunk_size": args.chunk_size,
                    "experimental_environment": {k: v for k, v in os.environ.items()
                                                  if k.startswith("INFERX_EXPERIMENTAL_")},
                    "binary_sha256": digest(args.binary),
                    "packages": {p: importlib.metadata.version(p) for p in ("torch", "transformers", "numpy")},
                    "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                    "scope": "single-sequence eager replay; no scheduler or graph coverage"}
        (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2))
        (args.output / "source.patch").write_bytes(subprocess.check_output(["git", "diff"], cwd=ROOT))
        with tarfile.open(args.output / "source.tar.gz", "w:gz") as archive:
            for folder in ("src", "include", "apps", "tests"):
                for path in sorted((ROOT / folder).rglob("*")):
                    if path.is_file() and path.suffix in (".h", ".cc", ".cu", ".txt", ".cmake"):
                        archive.add(path, arcname=str(path.relative_to(ROOT)))
            for path in sorted(Path(__file__).parent.glob("*.py")):
                archive.add(path, arcname=str(path.relative_to(ROOT)))
            archive.add(ROOT / "CMakeLists.txt", arcname="CMakeLists.txt")
        reference_path, candidate_path = args.output / "reference.f32", args.output / "inferx.f32"
        reference_logits(args.model, fixture, args.chunk_size, args.device, reference_path)
        # The reference model has been released; also return its cached GPU
        # allocations before starting the separate InferX process.
        if args.device == "cuda":
            import torch
            torch.cuda.empty_cache()
        env = os.environ.copy()
        env.pop("INFERX_EXPERIMENTAL_FLASH_ATTENTION", None)
        subprocess.run([str(args.binary.resolve()), "diagnostic", "replay-logits", "--model", str(args.model), "--fixture", str(args.fixture),
                        "--output", str(candidate_path), "--chunk-size", str(args.chunk_size)],
                       env=env, check=True)
        arrays = []
        for path in (reference_path, candidate_path):
            if path.stat().st_size != len(cases) * vocab * 4:
                raise ValueError("Incomplete logit dump")
            arrays.append(np.memmap(path, dtype="<f4", mode="r", shape=(len(cases), vocab)))
        rows = compare_logits(*arrays, args.atol, args.rtol)
        for case, row in zip(cases, rows):
            row.update(id=case["id"], archived_reference_token=case["reference_token"],
                       exact_archived_token=row["candidate_argmax"] == case["reference_token"])
        passed = all(r["numerical_pass"] and r["exact_argmax"] and r["exact_archived_token"] for r in rows)
        report = {"passed": passed, "atol": args.atol, "rtol": args.rtol, "rows": rows}
        (args.output / "comparison.json").write_text(json.dumps(report, indent=2, allow_nan=False))
        print(f"{'PASS' if passed else 'FAIL'}: {len(rows)} prefixes; report: {args.output / 'comparison.json'}")
        if not passed:
            raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("prepare")
    p.add_argument("--workload", type=Path, default=ROOT / "benchmarks/qwen3/workload.json")
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--candidate", type=Path, required=True)
    p.add_argument("--case", help="Optional explicit diagnostic subset; default is every case/repeat/request")
    p.add_argument("--output", type=Path, required=True)
    p.set_defaults(func=prepare)
    p = commands.add_parser("run")
    p.add_argument("--fixture", type=Path, required=True)
    p.add_argument("--model", type=Path, default=ROOT / "models/Qwen3-0.6B")
    p.add_argument("--binary", type=Path, default=ROOT / "build-cuda13/apps/inferx")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--chunk-size", type=int, default=4096)
    p.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    p.add_argument("--flash-attention", action="store_true",
                   help="Deprecated compatibility flag; FlashInfer is now the default")
    # No inferred/default full-model tolerance: the operator's 0.005 is not a logit contract.
    p.add_argument("--atol", type=float, required=True)
    p.add_argument("--rtol", type=float, required=True)
    p.set_defaults(func=run)
    args = parser.parse_args()
    if args.command == "run" and args.chunk_size <= 0:
        parser.error("--chunk-size must be positive")
    args.func(args)


if __name__ == "__main__":
    main()
