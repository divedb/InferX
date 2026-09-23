"""Serving benchmark over real HTTP servers (`vllm bench serve` as client).

Separation of concerns, following the vLLM/SGLang serving benchmarks: this
module owns server lifecycle (per-engine launchers, readiness, graceful
stop), load generation (the `vllm bench serve` CLI with a random dataset
and pinned knobs), the greedy agreement probe, and tag comparison. The
caller supplies a ModelConfig and a matrix of (input_len, output_len,
concurrency, num_prompts) cells — nothing here is model-specific.
"""
import argparse
import fcntl
import json
import os
import signal
import subprocess
import time
import urllib.request
from pathlib import Path

from models import ModelConfig
from provenance import digest, git_head, package_versions

AGREEMENT_PROMPTS = [
    "The capital of France is",
    "Water boils at a temperature of",
    "The first three prime numbers are",
    "In one sentence, photosynthesis is",
    "The largest planet in the solar system is",
    "A classic example of a prime number after 10 is",
    "The chemical formula of table salt is",
    "Cross-entropy loss in machine learning measures",
]


def environment():
    """Child environment; no_proxy keeps the HTTP client off any proxy."""
    env = os.environ.copy()
    bypass = env.get("no_proxy", env.get("NO_PROXY", ""))
    env["no_proxy"] = env["NO_PROXY"] = bypass + ",127.0.0.1,localhost"
    env["LD_LIBRARY_PATH"] = "/usr/lib/wsl/lib:" + env.get("LD_LIBRARY_PATH", "")
    env.update(CUDA_HOME="/usr/local/cuda", CUDA_VISIBLE_DEVICES="0",
               TOKENIZERS_PARALLELISM="false", VLLM_USE_V2_MODEL_RUNNER="0")
    env["PATH"] = "/usr/local/cuda/bin:" + env["PATH"]
    return env


def engine_command(engine: str, model: ModelConfig, port: int, vllm_bin: str,
                   inferx_bin: str, max_num_seqs: int, token_budget: int,
                   block_size: int, kv_bytes: int) -> list:
    """Launch command with pinned fairness conditions (server.md section 9):
    identical capacity knobs, greedy determinism seeds, no prefix caching."""
    if engine == "inferx":
        cmd = [inferx_bin, "serve", "--host", "127.0.0.1", "--port", str(port),
               "--model", model.path,
               "--max-num-seqs", str(max_num_seqs),
               "--max-num-batched-tokens", str(token_budget),
               "--block-size", str(block_size),
               "--kv-cache-memory-bytes", str(kv_bytes)]
    else:
        cmd = [vllm_bin, "serve", model.path, "--host", "127.0.0.1",
               "--port", str(port), "--dtype", model.dtype,
               "--seed", str(model.seed),
               "--max-model-len", str(model.max_model_len),
               "--generation-config", "vllm",
               "--block-size", str(block_size), "--no-enable-prefix-caching",
               "--enable-chunked-prefill", "--enforce-eager",
               "--max-num-seqs", str(max_num_seqs),
               "--max-num-batched-tokens", str(token_budget),
               "--gpu-memory-utilization", str(model.gpu_memory_utilization),
               "--kv-cache-memory-bytes", str(kv_bytes)]
    return cmd + model.engine_flags.get(engine, [])


def wait_ready(port, timeout_s, proc, log_path):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited early; see {log_path}")
        try:
            with urllib.request.urlopen(
                    f"http://127.0.0.1:{port}/health", timeout=1) as r:
                if r.status == 200:
                    return
        except Exception:
            time.sleep(2)
    raise RuntimeError(f"server not ready within {timeout_s}s; see {log_path}")


def stop_server(proc):
    if proc.poll() is not None:
        return
    proc.send_signal(signal.SIGINT)
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


def run_cell(vllm_bin: str, port: int, cell, out_dir: Path, model_name: str,
             cwd) -> Path:
    input_len, output_len, concurrency, num_prompts = cell
    name = f"i{input_len}-o{output_len}-c{concurrency}"
    result_path = out_dir / f"{name}.json"
    cmd = [vllm_bin, "bench", "serve", "--host", "127.0.0.1",
           "--port", str(port), "--model", model_name, "--backend", "openai",
           "--dataset-name", "random", "--seed", "0",
           "--random-input-len", str(input_len),
           "--random-output-len", str(output_len),
           "--num-prompts", str(num_prompts), "--temperature", "0",
           "--ignore-eos", "--max-concurrency", str(concurrency),
           "--percentile-metrics", "ttft,tpot,itl",
           "--save-result", "--result-dir", str(out_dir),
           "--result-filename", result_path.name]
    started = time.time()
    proc = subprocess.run(cmd, cwd=cwd, env=environment(),
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    (out_dir / f"{name}.bench.log").write_text(proc.stdout)
    if proc.returncode != 0:
        raise RuntimeError(f"bench failed for {name}; see {out_dir / (name + '.bench.log')}")
    print(f"  cell {name}: {time.time() - started:.1f}s", flush=True)
    return result_path


def agreement_probe(port, out_dir, engine, model_name):
    """Greedy agreement corpus: fixed text prompts, exact completions saved
    for the cross-engine diff. Complements the bench, which only measures."""
    texts = []
    for prompt in AGREEMENT_PROMPTS:
        body = json.dumps({"model": model_name, "prompt": prompt,
                           "max_tokens": 64, "temperature": 0,
                           "ignore_eos": True}).encode()
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/v1/completions", data=body,
            headers={"Content-Type": "application/json"})
        for attempt in range(3):
            try:
                with urllib.request.urlopen(req, timeout=120) as r:
                    texts.append(json.load(r)["choices"][0]["text"])
                break
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(2)
    path = out_dir / f"agree-{engine}.json"
    path.write_text(json.dumps(texts, indent=1))
    return path


def run_engine(root: Path, model: ModelConfig, engine: str, tag: str, matrix,
               results_root: Path, vllm_bin="python/.venv/bin/vllm",
               inferx_bin="build-cuda13/apps/inferx", port=8137,
               max_num_seqs=16, token_budget=4096, block_size=16, kv_bytes=3758096384):
    out_dir = Path(results_root) / tag
    out_dir.mkdir(parents=True, exist_ok=False)
    meta = dict(engine=engine, tag=tag, model=model.name,
                time=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                git_head=git_head(root),
                model_files={str(f): digest(f) for f in model.weights(root)},
                packages=package_versions(["torch", "vllm"]), matrix=matrix,
                fixed=dict(kv_bytes=kv_bytes, max_num_seqs=max_num_seqs,
                           token_budget=token_budget, block_size=block_size,
                           temperature=0, ignore_eos=True, dataset_seed=0,
                           vllm_flags="enforce-eager, no prefix caching, "
                                      "chunked prefill, explicit kv bytes"))
    log_path = out_dir / "server.log"
    with log_path.open("w") as log:
        cmd = engine_command(engine, model, port, str(root / vllm_bin),
                             str(root / inferx_bin), max_num_seqs, token_budget,
                             block_size, kv_bytes)
        meta["server_command"] = cmd
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT,
                                cwd=root, env=environment())
        try:
            wait_ready(port, 600 if engine == "vllm" else 120, proc, log_path)
            for cell in matrix:
                run_cell(str(root / vllm_bin), port, cell, out_dir, model.path,
                         cwd=root)
            meta["agreement"] = str(agreement_probe(port, out_dir, engine, model.path))
        finally:
            stop_server(proc)
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"{engine}: results in {out_dir}", flush=True)


def compare(results_root: Path, tags):
    def load(tag):
        base = Path(results_root) / tag
        cells = {path.stem: json.loads(path.read_text())
                 for path in sorted(base.glob("i*-o*-c*.json"))}
        return base, cells

    print(f"{'cell':<16}{'metric':<10}", "  ".join(f"{t:<16}" for t in tags))
    bases = {t: load(t) for t in tags}
    all_cells = sorted({c for _, cells in bases.values() for c in cells})
    for cell in all_cells:
        for key, label in [("mean_ttft_ms", "ttft_mean"), ("p99_ttft_ms", "ttft_p99"),
                           ("mean_tpot_ms", "tpot_mean"), ("mean_itl_ms", "itl_mean"),
                           ("p99_itl_ms", "itl_p99"),
                           ("output_throughput", "out_tok_s"),
                           ("request_success_rate", "success")]:
            row = f"{cell:<16}{label:<10}"
            for tag in tags:
                d = bases[tag][1].get(cell)
                value = "-" if d is None else d.get(key)
                row += f"  {str(value):<16}"
            print(row)
        print()
    texts = {}
    for tag, (base, _) in bases.items():
        candidates = sorted(base.glob("agree-*.json"))
        if candidates:
            texts[tag] = json.loads(candidates[0].read_text())
    if len(texts) == len(tags) == 2:
        a, b = tags
        same = sum(1 for x, y in zip(texts[a], texts[b]) if x == y)
        print(f"greedy agreement: {same}/{len(texts[a])} identical completions")
        for i, (x, y) in enumerate(zip(texts[a], texts[b])):
            if x != y:
                print(f"  first divergence at prompt {i}")
                break


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--engine", choices=["inferx", "vllm"])
    p.add_argument("--tag")
    p.add_argument("--model", required=True)
    p.add_argument("--cells",
                   help="Comma list of input_len:output_len:concurrency:num_prompts")
    p.add_argument("--results-root", type=Path, required=True)
    p.add_argument("--port", type=int, default=8137)
    p.add_argument("--compare", nargs="+")
    args = p.parse_args()
    if args.compare:
        compare(args.results_root, args.compare)
        return
    if not (args.engine and args.tag and args.cells):
        p.error("--engine, --tag and --cells are required (or --compare tags)")
    matrix = [tuple(int(v) for v in cell.split(":")) for cell in args.cells.split(",")]
    lock = open("/tmp/inferx-benchmark.lock", "w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    from models import get_model
    from pathlib import Path as P
    root = P(__file__).resolve().parents[2]
    run_engine(root, get_model(args.model), args.engine, args.tag, matrix,
               args.results_root, port=args.port)


if __name__ == "__main__":
    main()
