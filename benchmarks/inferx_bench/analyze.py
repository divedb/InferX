"""Compare archived run tags: summaries, agreement, CSV/JSON, plots.

Generic over model and suite: the caller supplies the results root (where
<tag>/b*.json and metadata.json live), the workload suite used by those
runs, and the model's vocab bound. Fail-closed: metadata mismatches across
compared tags, missing trials, or malformed rows abort the analysis before
any number is printed or plotted.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from metrics import trial_metrics
from validate import validate_metadata, validate_trials


def load_tag(results_root: Path, tag: str) -> list:
    path = results_root / tag
    rows = []
    for f in sorted(path.glob('b*.json')):
        if '.memory.' in f.name:
            continue
        rows.extend(json.loads(f.read_text()))
    return rows


def analyze(results_root: Path, suite_path: Path, tags, vocab_size=0,
            output_dir: Path = None, require_exact=False, plot_title='') -> dict:
    output_dir = output_dir or results_root
    output_dir.mkdir(parents=True, exist_ok=True)
    suite = json.loads(Path(suite_path).read_text())
    cases = suite['cases']
    workload_bytes = Path(suite_path).read_bytes()
    allrows = {t: load_tag(results_root, t) for t in tags}
    reference_meta = json.loads((results_root / tags[0] / 'metadata.json').read_text())

    summaries, flat = {}, []
    for tag, rows in allrows.items():
        meta = json.loads((results_root / tag / 'metadata.json').read_text())
        validate_metadata(meta, reference_meta, workload_bytes)
        validate_trials(rows, cases, meta['repeats'], vocab_size)
        summaries[tag] = {}
        for c in cases:
            trials = [r for r in rows if r['case'] == c['id']]
            assert len(trials) == meta['repeats']
            values = [trial_metrics(r, c) for r in trials]
            s = {k: float(np.median([v[k] for v in values])) for k in values[0]}
            s['e2e_min_tokens_s'] = min(v['e2e_tokens_s'] for v in values)
            s['e2e_max_tokens_s'] = max(v['e2e_tokens_s'] for v in values)
            s['repeat_deterministic'] = all(t['outputs'] == trials[0]['outputs'] for t in trials)
            summaries[tag][c['id']] = s
            flat.append(dict(tag=tag, case=c['id'], **s))

    with (output_dir / 'summary.csv').open('w') as f:
        w = csv.DictWriter(f, fieldnames=flat[0])
        w.writeheader()
        w.writerows(flat)

    reference = allrows[tags[0]]
    agreement = {}
    for tag, rows in allrows.items():
        agreement[tag] = {}
        for c in cases:
            ref = next(r for r in reference if r['case'] == c['id'])['outputs']
            got = next(r for r in rows if r['case'] == c['id'])['outputs']
            assert len(got) == len(ref) and all(len(x) == len(y) for x, y in zip(got, ref))
            match = sum(x == y for a, b in zip(got, ref) for x, y in zip(a, b))
            total = sum(map(len, ref))
            paired = [(r, next(v for v in reference if v['case'] == r['case'] and v['repeat'] == r['repeat']))
                      for r in rows if r['case'] == c['id']]
            agreement[tag][c['id']] = dict(
                exact_all_trials=all(r['outputs'] == v['outputs'] for r, v in paired),
                exact_sequences_all_trials=sum(a == b for r, v in paired for a, b in zip(r['outputs'], v['outputs'])),
                sequences_all_trials=sum(len(r['outputs']) for r, v in paired),
                matching_tokens=match, total_tokens=total,
                exact_sequences=sum(a == b for a, b in zip(got, ref)), sequences=len(ref),
                first_mismatch=[next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), None)
                                for a, b in zip(got, ref)])
    (output_dir / 'agreement.json').write_text(json.dumps(agreement, indent=2))
    (output_dir / 'summary.json').write_text(json.dumps(summaries, indent=2))

    plots = [('e2e_tokens_s', 'End-to-end output throughput', 'tokens/s'),
             ('decode_tokens_s', 'Decode window throughput', 'tokens/s'),
             ('prefill_tokens_s', 'Effective prefill throughput (includes queueing)', 'input tokens/s'),
             ('ttft_ms', 'Mean time to first token', 'ms'), ('itl_ms', 'Mean inter-token latency', 'ms'),
             ('peak_device_delta_mib', 'Sampled peak GPU memory above idle baseline', 'MiB')]
    fig, axes = plt.subplots(3, 2, figsize=(20, 15))
    x = np.arange(len(cases))
    width = .8 / len(tags)
    for ax, (key, title, unit) in zip(axes.flat, plots):
        for j, tag in enumerate(tags):
            vals = [summaries[tag][c['id']][key] for c in cases]
            errors = None
            if key == 'e2e_tokens_s':
                errors = [[v - summaries[tag][c['id']]['e2e_min_tokens_s'] for v, c in zip(vals, cases)],
                          [summaries[tag][c['id']]['e2e_max_tokens_s'] - v for v, c in zip(vals, cases)]]
            ax.bar(x + (j - (len(tags) - 1) / 2) * width, vals, width, label=tag,
                   yerr=errors, error_kw={'elinewidth': .6, 'capsize': 1})
        ax.set_xticks(x)
        ax.set_xticklabels([c['id'] for c in cases], rotation=80, fontsize=7)
        ax.set_title(title)
        ax.set_ylabel(unit)
        ax.grid(axis='y', alpha=.2)
        ax.legend(fontsize=8)
    fig.suptitle(plot_title + '\nNumerical agreement: see agreement.json. '
                 'Memory: sampled whole-device NVML, including startup')
    fig.tight_layout()
    fig.savefig(output_dir / 'comparison.png', dpi=160)
    fig.savefig(output_dir / 'comparison.pdf')

    for tag in tags[1:]:
        ratios = [summaries[tag][c['id']]['e2e_tokens_s'] /
                  summaries[tags[0]][c['id']]['e2e_tokens_s'] for c in cases]
        print(tag, 'speedup range', min(ratios), max(ratios),
              'geomean', np.exp(np.mean(np.log(ratios))))
        print('exact sequences (first trial)',
              sum(a['exact_sequences'] for a in agreement[tag].values()), '/',
              sum(a['sequences'] for a in agreement[tag].values()))
    if require_exact and not all(a['exact_all_trials'] for tag in tags for a in agreement[tag].values()):
        raise SystemExit('FAIL: greedy equivalence not established; '
                         'artifacts saved, performance is diagnostic only.')
    return dict(summaries=summaries, agreement=agreement)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('tags', nargs='+')
    p.add_argument('--results-root', type=Path, required=True,
                   help='Directory holding <tag>/ run archives')
    p.add_argument('--suite', type=Path, required=True, help='Workload suite JSON')
    p.add_argument('--vocab-size', type=int, default=0,
                   help='Token-ID upper bound for validation (0 skips the bound)')
    p.add_argument('--require-exact', action='store_true',
                   help='Exit nonzero after saving artifacts if any greedy output differs')
    p.add_argument('--output-dir', type=Path, default=None,
                   help='Keep new comparisons separate from historical reports')
    p.add_argument('--plot-title', default='')
    args = p.parse_args()
    analyze(args.results_root, args.suite, args.tags, args.vocab_size,
            args.output_dir, args.require_exact, args.plot_title)


if __name__ == '__main__':
    main()
