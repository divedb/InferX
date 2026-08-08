#!/usr/bin/env python3
"""Generate reference logits for DeepSeek-V2-Lite — the §18.7 D5 oracle.

The reference is transformers' native DeepseekV2ForCausalLM, sharing nothing
with the C++ side but the weights. Native rather than the checkpoint's bundled
modeling code (trust_remote_code): the bundled file is frozen against a 2024
transformers API and no longer imports on current versions, while the native
class is maintained -- and measured against the same weights they are the same
model. Agreement means the MLA projections, the decoupled-RoPE
convention, the YaRN frequency table and mscale softmax scale, the dense/MoE
layer schedule, softmax/greedy routing and the ungated shared experts all
match the definition rather than one reading of it.

Precision: run bf16 on a GPU that fits ~31.4 GB of weights (the rented-box
case this script exists for). With --fp32 and ~70 GB of host RAM it will run
fp32 on CPU instead — slower by hours, but then the reference is strictly
more precise than the implementation and every disagreement is ours, the way
the Qwen2.5 goldens work.

The one caveat carried over from gpt-oss: V2-Lite routes each token to 6 of
64 experts, and where the router's 6th/7th margin is within bf16 noise, an
independent implementation can route differently and diverge legitimately.
Short prompts rarely hit that; the argmax gate in
deepseek_v2_reference_test.cc is still the property to lean on.

Output is the same 'IXRL' container as gen_reference_logits.py:

    magic   'IXRL'          4 bytes
    version u32 = 1
    tokens  u32
    vocab   u32
    ids     i32 * tokens
    logits  f32 * tokens * vocab   (row-major, fp32)

Usage:
    scripts/gen_deepseek_logits.py <checkpoint-dir> <output.bin> [--prompt TEXT]
    scripts/gen_deepseek_logits.py <checkpoint-dir> <output.bin> [--fp32] [id ...]
"""

import struct
import sys

MAGIC = b"IXRL"
VERSION = 1

DEFAULT_PROMPT = "The capital of France is"


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__)
        return 2

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    checkpoint = sys.argv[1]
    output = sys.argv[2]
    rest = sys.argv[3:]

    fp32 = "--fp32" in rest
    rest = [a for a in rest if a != "--fp32"]

    tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=False)

    if rest and rest[0] == "--prompt":
        ids = tokenizer(" ".join(rest[1:]) or DEFAULT_PROMPT)["input_ids"]
    elif rest:
        ids = [int(a) for a in rest]
    else:
        ids = tokenizer(DEFAULT_PROMPT)["input_ids"]

    if fp32:
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint, torch_dtype=torch.float32, trust_remote_code=False
        )
        device = "cpu"
    else:
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint, torch_dtype=torch.bfloat16, trust_remote_code=False,
            device_map="cuda",
        )
        device = "cuda"
    model.eval()

    with torch.no_grad():
        input_ids = torch.tensor([ids], dtype=torch.long, device=device)
        logits = model(input_ids).logits[0].float().cpu()

    tokens, vocab = logits.shape
    assert tokens == len(ids)

    with open(output, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<III", VERSION, tokens, vocab))
        f.write(struct.pack(f"<{tokens}i", *ids))
        f.write(logits.numpy().astype("<f4").tobytes())

    print(f"wrote {tokens} x {vocab} logits for ids {ids} to {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
