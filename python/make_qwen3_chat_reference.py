"""Regenerate native chat renderer goldens using the local HF checkpoint.

Run from the repository root with python/.venv/bin/python. Existing message
cases and thinking options are preserved; inspect changes before accepting
updated reference output or changing the recognized native template.
"""
import json
from pathlib import Path

from transformers import AutoTokenizer


def main():
    tokenizer = AutoTokenizer.from_pretrained("models/Qwen3-0.6B", local_files_only=True)
    path = Path("tests/testdata/qwen3_chat_reference.json")
    cases = json.loads(path.read_text())
    for case in cases:
        case["rendered"] = tokenizer.apply_chat_template(
            case["messages"], tokenize=False, add_generation_prompt=True,
            enable_thinking=case["enable_thinking"],
        )
    path.write_text(json.dumps(cases, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
