"""Latency/throughput metrics computed from one raw trial row.

A trial row is the JSON the engines emit per case+repeat: per-request token
arrival timestamps (`arrivals_ms`, ms since submission), the token IDs
(`outputs`), the wall-clock span (`e2e_ms`), and NVML-sampled peak device
memory. Metric definitions are service-level (they include scheduling and
queueing, by design — see the qwen3 README for the caveats) and are
identical for every engine, which is what makes cross-engine comparison
meaningful.
"""
import numpy as np


def trial_metrics(row, case) -> dict:
    """One trial -> the metric dictionary used in summaries and CSV rows.

    Case context supplies the request count and token lengths; explicit
    counts and batch/concurrency are embedded so every summary row is
    self-describing without joining against the workload file.
    """
    arrivals = row['arrivals_ms']
    outputs = row['outputs']
    n, k = case['concurrency'], case['output_len']
    assert len(arrivals) == len(outputs) == n
    assert all(len(x) == k for x in arrivals + outputs)
    assert all(all(y >= x for x, y in zip(r, r[1:])) for r in arrivals)

    first = np.array([x[0] for x in arrivals])
    last = max(x[-1] for x in arrivals)
    itl = np.concatenate([np.diff(x) for x in arrivals])
    return dict(
        # Explicit configuration echo: counts, lengths, concurrency.
        input_tokens=n * case['input_len'],
        output_tokens=n * k,
        batch=case['batch'],
        concurrency=n,
        # Throughput.
        e2e_tokens_s=n * k * 1000 / row['e2e_ms'],
        decode_tokens_s=n * (k - 1) * 1000 / (last - min(first)),
        prefill_tokens_s=n * case['input_len'] * 1000 / max(first),
        # Latency.
        e2e_ms=row['e2e_ms'],
        ttft_ms=float(np.mean(first)), ttft_p95_ms=float(np.percentile(first, 95)),
        itl_ms=float(np.mean(itl)), itl_p95_ms=float(np.percentile(itl, 95)),
        # Memory (whole-device NVML samples; see README for caveats).
        peak_device_mib=row['peak_device_mib'],
        peak_device_delta_mib=row['peak_device_delta_mib'],
    )
