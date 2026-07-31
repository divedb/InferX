#!/usr/bin/env python3
"""Generate reference logits from HuggingFace transformers for M2's conformance test.

This is the independent implementation our forward pass is checked against. It
deliberately shares nothing with the C++ side except the checkpoint on disk: it
runs HF's own Qwen2ForCausalLM, so agreement means our layer order, RoPE
convention, GQA mapping, norm placement and tied-embedding handling all match
the definition rather than matching my reading of it.

Runs on CPU in float32. That is slow and completely fine -- it is a handful of
tokens, once -- and fp32 makes the reference cleaner than the bf16 stack it is
being compared to, so any disagreement is attributable to us rather than shared
between us.

Output is a flat binary file:

    magic   'IXRL'          4 bytes
    version u32 = 1
    tokens  u32
    vocab   u32
    ids     i32 * tokens
    logits  f32 * tokens * vocab   (row-major, fp32)

Usage:
    scripts/gen_reference_logits.py <checkpoint-dir> <output.bin> [id ...]
"""

import struct
import sys

import torch
from transformers import AutoModelForCausalLM

MAGIC = b"IXRL"
VERSION = 1

# "The capital of France is" in Qwen2.5's vocabulary. Kept in sync with
# tests/kernel/qwen2_reference_test.cc by construction: the ids are written into
# the output file and the test reads them from there rather than restating them.
DEFAULT_IDS = [785, 6722, 315, 9625, 374]


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__)
        return 2

    ckpt_dir = sys.argv[1]
    out_path = sys.argv[2]

    # Optional leading --dtype. Generating the *same* logits in bf16 as well as
    # fp32 is what makes the tolerance in the C++ test derivable rather than
    # invented: the gap between HF-fp32 and HF-bf16 is the error bf16 alone
    # introduces, which is the budget our bf16 stack is entitled to.
    rest = sys.argv[3:]
    dtype = torch.float32
    if rest and rest[0] == "--dtype":
        dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16}[rest[1]]
        rest = rest[2:]

    ids = [int(a) for a in rest] or DEFAULT_IDS

    print(f"loading {ckpt_dir} on cpu in {dtype} ...", flush=True)
    # No device_map: that path wants `accelerate`, and this is a CPU-only torch
    # build where CPU is already the default.
    model = AutoModelForCausalLM.from_pretrained(ckpt_dir, dtype=dtype)
    model.eval()

    print(f"running {len(ids)} tokens: {ids}", flush=True)
    with torch.no_grad():
        out = model(input_ids=torch.tensor([ids], dtype=torch.long))

    # [1, tokens, vocab] -> [tokens, vocab]
    logits = out.logits[0].to(torch.float32).contiguous()
    tokens, vocab = logits.shape
    print(f"logits: {tokens} x {vocab}")

    top = int(torch.argmax(logits[-1]))
    print(f"reference argmax at last position: {top}")

    with open(out_path, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<III", VERSION, tokens, vocab))
        f.write(struct.pack(f"<{len(ids)}i", *ids))
        f.write(logits.numpy().astype("<f4").tobytes())

    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
