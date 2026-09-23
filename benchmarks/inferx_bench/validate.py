"""Fail-closed validation for archived benchmark data (no GPU required).

Generic over model and workload: the token-ID upper bound comes from the
model configuration rather than being hardcoded, and every structural
invariant (case coverage, output lengths, timestamp monotonicity, memory
readings) is enforced before any number is believed.
"""
import hashlib
import math


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_metadata(metadata, reference, workload_bytes):
    require(metadata['workload_sha256'] == hashlib.sha256(workload_bytes).hexdigest(),
            'Workload hash differs from the analyzed workload')
    for field in ('weights', 'config_sha256', 'gpu', 'driver', 'packages', 'platform', 'repeats'):
        require(metadata[field] == reference[field], f'Comparison metadata mismatch: {field}')
    require(bool(metadata['weights']), 'Missing checkpoint hashes')
    require(metadata['repeats'] > 0, 'No measured repeats')


def validate_trials(rows, cases, repeats, vocab_size=0):
    expected = {(c['id'], repeat) for c in cases for repeat in range(repeats)}
    keys = [(r['case'], r['repeat']) for r in rows]
    require(len(keys) == len(set(keys)), 'Duplicate case/repeat pair')
    require(set(keys) == expected, 'Missing or unexpected measured trials')
    by_id = {c['id']: c for c in cases}
    for r in rows:
        c = by_id[r['case']]
        arrivals, outputs = r['arrivals_ms'], r['outputs']
        require(len(arrivals) == len(outputs) == c['concurrency'], 'Wrong request count')
        require(math.isfinite(r['e2e_ms']) and r['e2e_ms'] > 0, 'Invalid end-to-end duration')
        for times, tokens in zip(arrivals, outputs):
            require(len(times) == len(tokens) == c['output_len'], 'Wrong output length')
            if vocab_size:
                require(all(isinstance(t, int) and 0 <= t < vocab_size for t in tokens),
                        'Invalid token ID')
            require(all(math.isfinite(t) and 0 < t <= r['e2e_ms'] for t in times),
                    'Invalid token arrival timestamp')
            require(all(y >= x for x, y in zip(times, times[1:])), 'Nonmonotonic arrivals')
            require(times[-1] > times[0], 'No measured decode interval')
        for key in ('peak_device_mib', 'peak_device_delta_mib'):
            require(math.isfinite(r[key]) and r[key] >= 0, 'Invalid memory measurement')
