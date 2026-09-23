"""Freeze the workload before optimization; no tokenizer in measured paths."""
import hashlib
import json
from pathlib import Path
base = json.loads(Path('benchmarks/prompts.json').read_text())
cases = []
for length in (128, 512, 1024):
    for output in (32, 128):
        for batch, concurrency in ((1, 1), (4, 4), (4, 8), (16, 16)):
            prompts = [(p * ((length + len(p)-1)//len(p)))[:length] for p in base[:concurrency]]
            cases.append(dict(id=f'p{length}_o{output}_b{batch}_c{concurrency}',
                              input_len=length, output_len=output, batch=batch,
                              concurrency=concurrency, prompts=prompts))
workload = dict(token_budget=4096, kv_blocks=2048, block_size=16, cases=cases,
                sampling=dict(temperature=0, top_p=1, top_k=-1, ignore_eos=True,
                              repetition_penalty=1, presence_penalty=0, frequency_penalty=0),
                provenance='First N existing benchmarks/prompts.json prompts; truncate or repeat token IDs to exact length.')
p = Path('benchmarks/qwen3/workload.json')
p.write_text(json.dumps(workload, separators=(',', ':'))+'\n')
print(hashlib.sha256(p.read_bytes()).hexdigest())
