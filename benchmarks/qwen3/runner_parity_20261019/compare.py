"""Compare measured token IDs from the two archived workload runs."""
import hashlib
import importlib.metadata
import json
from pathlib import Path
from tokenizers import Tokenizer

out = Path(__file__).resolve().parent
root = out.parents[2]
suite = json.loads((out / 'workload.json').read_text())
tokenizer = Tokenizer.from_file(str(root / 'models/Qwen3-0.6B/tokenizer.json'))
records = {}
for engine in ['inferx', 'vllm']:
    records[engine] = {}
    for batch in [1, 4, 16]:
        for line in (out / f'{engine}_b{batch}.log').read_text().splitlines():
            if not line.startswith('{'):
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get('engine') == engine and 'outputs' in row:
                key = (row['case'], row['repeat'])
                assert key not in records[engine], key
                records[engine][key] = row
    assert len(records[engine]) == len(suite['cases'])
report = {'cases': [], 'exact_sequences': 0, 'sequences': 0,
          'matching_token_positions': 0, 'token_positions': 0, 'mismatches': []}
for case in suite['cases']:
    a = records['inferx'][(case['id'], 0)]['outputs']
    b = records['vllm'][(case['id'], 0)]['outputs']
    assert len(a) == len(b) == case['concurrency']
    entry = {'case': case['id'], 'batch': case['batch'], 'exact_sequences': 0,
             'sequences': len(a), 'matching_token_positions': 0, 'token_positions': 0}
    for i, (left, right) in enumerate(zip(a, b)):
        assert len(left) == len(right) == case['output_len']
        entry['exact_sequences'] += left == right
        entry['matching_token_positions'] += sum(x == y for x, y in zip(left, right))
        entry['token_positions'] += len(left)
        if left != right:
            first = next(j for j, (x, y) in enumerate(zip(left, right)) if x != y)
            report['mismatches'].append({'case': case['id'], 'sequence': i,
                'first_different_generated_token_index': first,
                'inferx_token': left[first], 'vllm_token': right[first],
                'inferx_token_text': tokenizer.decode([left[first]], skip_special_tokens=False),
                'vllm_token_text': tokenizer.decode([right[first]], skip_special_tokens=False),
                'common_generated_prefix': tokenizer.decode(left[:first], skip_special_tokens=False),
                'inferx_output': tokenizer.decode(left, skip_special_tokens=False),
                'vllm_output': tokenizer.decode(right, skip_special_tokens=False)})
    report['cases'].append(entry)
    for key in ['exact_sequences', 'sequences', 'matching_token_positions', 'token_positions']:
        report[key] += entry[key]
(out / 'comparison.json').write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
meta = json.loads((out / 'metadata.json').read_text())
meta['packages'] = {p: importlib.metadata.version(p) for p in ['vllm', 'torch', 'transformers']}
meta['binary_sha256'] = hashlib.sha256((root / 'build-dbg/apps/inferx').read_bytes()).hexdigest()
meta['weights_sha256'] = hashlib.file_digest((root / 'models/Qwen3-0.6B/model.safetensors').open('rb'), 'sha256').hexdigest()
(out / 'metadata.json').write_text(json.dumps(meta, indent=2) + '\n')
print(json.dumps({k: v for k, v in report.items() if k not in ['cases', 'mismatches']}, indent=2))
for batch in [1, 4, 16]:
    rows = [c for c in report['cases'] if c['batch'] == batch]
    print('batch', batch, 'exact', sum(c['exact_sequences'] for c in rows), '/', sum(c['sequences'] for c in rows))
for mismatch in report['mismatches'][:4]:
    print(json.dumps({k:v for k,v in mismatch.items() if not k.endswith('_output')}, ensure_ascii=False))
