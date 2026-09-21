#!/usr/bin/env python3
"""Generates the Qwen3-0.6B greedy reference tokens for engine_e2e_test.

Runs the checkpoint through HuggingFace Transformers with greedy decoding
and writes prompt/expected token ids under tests/testdata/qwen3_0_6b/.
Run with the project venv: python/.venv/bin/python python/make_qwen3_reference.py
"""

import argparse
import pathlib

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROMPT = "The capital of France is"
MAX_NEW_TOKENS = 24


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/Qwen3-0.6B")
    parser.add_argument("--out", default="tests/testdata/qwen3_0_6b")
    parser.add_argument("--prompt", default=PROMPT)
    parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16
    ).to(device)
    model.eval()

    prompt_ids = tokenizer(args.prompt, return_tensors="pt").input_ids[0].tolist()
    with torch.no_grad():
        output = model.generate(
            torch.tensor([prompt_ids]).to(device),
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
            min_new_tokens=args.max_new_tokens,  # keep going past EOS
        )
    generated = output[0][len(prompt_ids):].tolist()

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "prompt_tokens.txt").write_text(",".join(map(str, prompt_ids)))
    (out / "expected_tokens.txt").write_text(",".join(map(str, generated)))
    print(f"prompt ({len(prompt_ids)} tokens): {prompt_ids}")
    print(f"expected ({len(generated)} tokens): {generated}")
    print(f"text: {tokenizer.decode(generated)}")


if __name__ == "__main__":
    main()
