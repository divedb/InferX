#!/usr/bin/env python3
"""Generate reference logits for gpt-oss-20b — the oracle Phase 0 exists to build.

The counterpart to gen_reference_logits.py, and it differs from it in one way
that is forced rather than chosen: **this reference runs in bf16 on the GPU, not
fp32 on the CPU.** gpt-oss's expert weights are MXFP4, and dequantizing them to
fp32 would be 76 GB. There is no fp32 reference to be had on any machine we
have, so the honest thing is to say what the reference is and set the tolerance
accordingly.

That has a consequence worth stating before anyone reads a tolerance off this
file. For Qwen2.5 the reference is more precise than the implementation, so a
disagreement is ours. Here the reference is the *same* precision as the
implementation and uses the same 4-bit weights, so a disagreement is ours only
if it exceeds what two different bf16 evaluation orders can produce. Argmax
agreement at every position is the property to lean on; elementwise closeness is
weaker evidence than it is for the dense path.

What makes it an oracle at all is that it shares nothing with the C++ side but
the checkpoint: it is HF's own GptOssForCausalLM, so agreement means our expert
routing, MXFP4 decode, attention sinks, sliding-window masking, YaRN and clamped
interleaved SwiGLU all match the definition rather than matching one reading of
it.

Requires, in .venv-vllm:  transformers, torch, kernels>=0.15.2,<0.16, accelerate
Without `kernels` in that window, transformers silently falls back to
dequantizing to bf16 — 38 GB, which will not load — so the version is asserted
below rather than discovered as an out-of-memory error twenty minutes in.

Output is byte-identical in format to gen_reference_logits.py, so the C++ side
reads both with one loader:

    magic   'IXRL'          4 bytes
    version u32 = 1
    tokens  u32
    vocab   u32
    ids     i32 * tokens
    logits  f32 * tokens * vocab   (row-major, fp32)

Usage:
    scripts/gen_gptoss_logits.py <checkpoint-dir> <output.bin> [--prompt TEXT]
    scripts/gen_gptoss_logits.py <checkpoint-dir> <output.bin> [id ...]
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
    from transformers.utils import is_kernels_available

    ckpt_dir = sys.argv[1]
    out_path = sys.argv[2]
    rest = sys.argv[3:]

    # The check that matters. Without a usable `kernels`, transformers falls
    # back to dequantizing MXFP4 to bf16 and the load dies at ~38 GB; the
    # warning it prints scrolls past inside a progress bar and the failure
    # arrives much later looking like something else.
    if not is_kernels_available():
        from transformers.utils.import_utils import (
            KERNELS_MAX_VERSION,
            KERNELS_MIN_VERSION,
        )
        print(
            f"error: transformers wants kernels>={KERNELS_MIN_VERSION},"
            f"<{KERNELS_MAX_VERSION} to run MXFP4 natively.\n"
            f"  pip install --no-deps 'kernels>={KERNELS_MIN_VERSION},"
            f"<{KERNELS_MAX_VERSION}' kernels_data accelerate\n"
            "Without it the model is dequantized to bf16 (~38 GB) and will not "
            "load.",
            file=sys.stderr,
        )
        return 1

    if not torch.cuda.is_available():
        print("error: no CUDA device; the MXFP4 path needs one", file=sys.stderr)
        return 1

    tok = AutoTokenizer.from_pretrained(ckpt_dir)

    if rest and rest[0] == "--prompt":
        ids = tok.encode(" ".join(rest[1:]))
    elif rest:
        ids = [int(a) for a in rest]
    else:
        ids = tok.encode(DEFAULT_PROMPT)

    print(f"loading {ckpt_dir} on cuda:0, MXFP4 native ...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        ckpt_dir, dtype="auto", device_map="cuda:0")
    model.eval()

    resident = torch.cuda.memory_allocated() / 1e9
    print(f"resident weights: {resident:.2f} GB", flush=True)

    print(f"running {len(ids)} tokens: {ids}", flush=True)
    print(f"  as text: {tok.decode(ids)!r}", flush=True)

    with torch.no_grad():
        out = model(input_ids=torch.tensor([ids], device="cuda:0"))

    logits = out.logits[0].to(torch.float32).cpu().contiguous()
    tokens, vocab = logits.shape
    print(f"logits: {tokens} x {vocab}")

    top = int(torch.argmax(logits[-1]))
    print(f"reference argmax at last position: {top} ({tok.decode([top])!r})")

    with open(out_path, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<III", VERSION, tokens, vocab))
        f.write(struct.pack(f"<{len(ids)}i", *ids))
        f.write(logits.numpy().astype("<f4").tobytes())

    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
