"""Workload-suite schema and loading.

A suite is a JSON document of raw token-ID prompts plus engine capacity
knobs; it deliberately contains no tokenizer and no model name, so one
frozen suite can drive any engine that accepts token IDs. Suits are hashed
into run metadata, so they are immutable once referenced by archived tags
(see the qwen3 README for the discipline).

Schema:
  token_budget, kv_blocks, block_size : engine capacity knobs
  sampling                            : greedy-by-default sampling block
  cases[]                             : id, input_len, output_len, batch,
                                        concurrency, prompts (token IDs)
"""
import json
from pathlib import Path


def load_suite(path) -> dict:
    suite = json.loads(Path(path).read_text())
    for key in ('token_budget', 'kv_blocks', 'block_size', 'cases', 'sampling'):
        if key not in suite:
            raise ValueError(f'workload suite {path} is missing "{key}"')
    for case in suite['cases']:
        for key in ('id', 'input_len', 'output_len', 'batch', 'concurrency', 'prompts'):
            if key not in case:
                raise ValueError(f'workload case is missing "{key}": {case}')
        if len(case['prompts']) != case['concurrency']:
            raise ValueError(f'case {case["id"]}: prompt count != concurrency')
    return suite


def cases_for_batch(suite: dict, batch: int) -> list:
    return [c for c in suite['cases'] if c['batch'] == batch]


def batches(suite: dict) -> list:
    return sorted({c['batch'] for c in suite['cases']})
