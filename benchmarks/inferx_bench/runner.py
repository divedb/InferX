"""Offline suite driver: run engines serially with full provenance.

One batch-size process at a time under an exclusive GPU lock, an NVML
monitor sampling whole-device memory around each child, fail-closed trial
validation, and a run directory that records commands, hashes, environment,
a source archive and the dirty diff. Nothing in here knows which model is
being measured beyond the ModelConfig it is handed.
"""
import fcntl
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import pynvml

from provenance import (digest, git_head, harness_hashes, package_versions,
                        platform, write_source_archive)
from validate import validate_trials
from workload import batches, cases_for_batch


def environment():
    """Pinned child environment; benchmark host specifics, not model ones."""
    env = os.environ.copy()
    env['LD_LIBRARY_PATH'] = '/usr/lib/wsl/lib:' + env.get('LD_LIBRARY_PATH', '')
    env.update(VLLM_USE_V2_MODEL_RUNNER='0', CUDA_HOME='/usr/local/cuda',
               CUDA_VISIBLE_DEVICES='0', TOKENIZERS_PARALLELISM='false')
    env['PATH'] = '/usr/local/cuda/bin:' + env['PATH']
    env.pop('INFERX_PROFILE_OPS', None)
    env.pop('INFERX_DIAGNOSTIC_TRACE_DIR', None)
    env.pop('INFERX_DIAGNOSTIC_TRACE_STEP', None)
    return env


def experimental_env(cuda_graphs=False, decode_linear=False, split_decode=False,
                     packed_projections=False, prefill_tile=64, scalar_kv=False):
    """InferX engine experiment switches (engine-generic, off by default)."""
    return {
        'INFERX_EXPERIMENTAL_DECODE_LINEAR': '1' if decode_linear else '0',
        'INFERX_EXPERIMENTAL_SPLIT_DECODE': '1' if split_decode else '0',
        'INFERX_EXPERIMENTAL_PACKED_PROJECTIONS': '1' if packed_projections else '0',
        'INFERX_EXPERIMENTAL_PREFILL_TILE': str(prefill_tile),
        'INFERX_DIAGNOSTIC_SCALAR_KV': '1' if scalar_kv else '0',
    }


def run_suite(root: Path, model, engine: str, tag: str, suite_path: str, repeats: int,
              results_root: Path, inferx_binary='build-cuda13/apps/inferx',
              cuda_graphs=False, decode_linear=False, split_decode=False,
              packed_projections=False, prefill_tile=64, scalar_kv=False,
              worker_module='benchmarks/inferx_bench/vllm_worker.py',
              harness_dirs=()):
    """Execute one engine over the suite; returns the run directory.

    `harness_dirs` are extra directories whose *.py files are hashed into
    metadata (model-specific wrappers), preserving the old per-model
    provenance shape.
    """
    root = Path(root)
    suite_path = str(suite_path)
    lock_file = open('/tmp/inferx-benchmark.lock', 'w')
    fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
    pynvml.nvmlInit()
    out = Path(results_root) / tag
    out.mkdir(parents=True, exist_ok=False)
    gpu = pynvml.nvmlDeviceGetHandleByIndex(0)
    env = environment()
    env.pop('INFERX_EXPERIMENTAL_FLASH_ATTENTION', None)
    env.update(experimental_env(cuda_graphs, decode_linear, split_decode,
                                packed_projections, prefill_tile, scalar_kv))
    from workload import load_suite
    suite = load_suite(suite_path)
    kv_bytes = model.kv_bytes(suite['kv_blocks'], suite['block_size'])
    meta = dict(
        engine=engine, tag=tag, model=model.name, time=time.strftime('%Y-%m-%dT%H:%M:%S%z'),
        git_head=git_head(root), workload_sha256=digest(suite_path),
        weights={str(f): digest(f) for f in model.weights(root)},
        config_sha256=digest(root / model.path / 'config.json'),
        gpu=pynvml.nvmlDeviceGetName(gpu), driver=pynvml.nvmlSystemGetDriverVersion(),
        packages=package_versions(['torch', 'vllm', 'numpy']),
        environment={k: v for k, v in env.items() if k in (
            'VLLM_USE_V2_MODEL_RUNNER', 'VLLM_ATTENTION_BACKEND',
            'VLLM_WORKER_MULTIPROC_METHOD', 'VLLM_USE_V1',
            'INFERX_EXPERIMENTAL_FLASH_ATTENTION', 'CUDA_VISIBLE_DEVICES',
            'CUDA_HOME', 'LD_LIBRARY_PATH', 'TOKENIZERS_PARALLELISM', 'OMP_NUM_THREADS')},
        attention_backend='flashinfer' if engine == 'inferx' else 'vllm-default',
        experimental_flags=dict(cuda_graphs=cuda_graphs, decode_linear=decode_linear,
                                split_decode=split_decode,
                                packed_projections=packed_projections,
                                prefill_tile=prefill_tile, scalar_kv=scalar_kv),
        kv_cache_bytes=kv_bytes,
        harness_sha256=harness_hashes(
            [root / 'benchmarks/inferx_bench', *harness_dirs]),
        platform=platform(),
        binary_sha256=digest(root / inferx_binary), repeats=repeats,
        memory_method='NVML total device used sampled every 5ms, including '
                      'initialization/warmup; baseline before child startup; '
                      'per-process accounting unavailable on WSL')
    (out / 'metadata.json').write_text(json.dumps(meta, indent=2))
    (out / 'source.patch').write_bytes(subprocess.check_output(['git', 'diff'], cwd=root))
    write_source_archive(root, out / 'source.tar.gz', extra_py_dirs=harness_dirs + (
        root / 'benchmarks/inferx_bench',))

    for batch in batches(suite):
        if engine == 'inferx' and digest(root / inferx_binary) != meta['binary_sha256']:
            raise RuntimeError('Benchmark binary changed between batch processes')
        cmd = ([sys.executable, worker_module] if engine == 'vllm'
               else [inferx_binary, 'bench', 'workload', '--attention-backend', 'flashinfer'])
        cmd += ['--max-num-seqs' if engine == 'inferx' else '--batch', str(batch),
                '--repeats', str(repeats), '--suite', suite_path]
        cmd += (['--model', model.path, '--kv-cache-memory-bytes', str(kv_bytes)]
                if engine == 'inferx' else
                ['--model', model.path, '--kv-bytes', str(kv_bytes)])
        if engine == 'inferx' and cuda_graphs:
            cmd += ['--cuda-graphs']
        baseline = pynvml.nvmlDeviceGetMemoryInfo(gpu).used
        samples = []
        stop = threading.Event()
        monitor_errors = []

        def monitor():
            try:
                while not stop.is_set():
                    samples.append((time.time(),
                                    pynvml.nvmlDeviceGetMemoryInfo(gpu).used,
                                    pynvml.nvmlDeviceGetUtilizationRates(gpu).gpu))
                    stop.wait(.005)
            except Exception as error:
                monitor_errors.append(str(error))

        thread = threading.Thread(target=monitor)
        thread.start()
        try:
            with (out / f'b{batch}.log').open('w') as log:
                result = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT,
                                        cwd=root, env=env)
        finally:
            stop.set()
            thread.join()
        if monitor_errors or not samples:
            raise RuntimeError(f'GPU memory monitor failed: {monitor_errors}')
        (out / f'b{batch}.memory.json').write_text(json.dumps(dict(
            baseline_bytes=baseline, peak_bytes=max(s[1] for s in samples),
            samples=samples, command=cmd)))
        if result.returncode:
            raise RuntimeError(f'{cmd} failed; inspect {out}/b{batch}.log')
        if engine == 'inferx' and digest(root / inferx_binary) != meta['binary_sha256']:
            raise RuntimeError('Benchmark binary changed during measurement')

        rows = []
        for line in (out / f'b{batch}.log').read_text().splitlines():
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if isinstance(r, dict) and 'arrivals_ms' in r:
                r['peak_device_mib'] = max(s[1] for s in samples) / 2**20
                r['peak_device_delta_mib'] = (max(s[1] for s in samples) - baseline) / 2**20
                rows.append(r)
        validate_trials(rows, cases_for_batch(suite, batch), repeats,
                        model.vocab_size)
        (out / f'b{batch}.json').write_text(json.dumps(rows))
        print(tag, 'batch', batch, 'complete:', len(rows), 'trials', flush=True)
    return out
