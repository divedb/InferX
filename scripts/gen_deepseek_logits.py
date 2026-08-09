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
    scripts/gen_deepseek_logits.py <checkpoint-dir> <output.bin> \
        --decode-output decode.bin --max-new-tokens 128 [--prompt TEXT]
"""

import struct
import sys

MAGIC = b"IXRL"
VERSION = 1

DEFAULT_PROMPT = "The capital of France is"


def fix_yarn_softmax_scale(model) -> None:
    """Restore the YaRN mscale^2 softmax-scale correction HF's port dropped.

    The checkpoint's own modeling_deepseek.py multiplies softmax_scale by
    yarn_get_mscale(factor, mscale_all_dim)^2 (and vLLM does the same), but
    transformers' integrated DeepseekV2Attention uses a bare
    qk_head_dim**-0.5 -- measured directly on transformers 5.14.1, where that
    one difference moved 17 of 65 teacher-forced argmaxes on a chat prompt.
    Without this patch the generated reference encodes the port's bug and no
    correct engine can match it.
    """
    import math

    config = model.config
    rope = (getattr(config, "rope_scaling", None)
            or getattr(config, "rope_parameters", None))
    if not rope or rope.get("rope_type", rope.get("type")) != "yarn":
        return
    mscale_all_dim = rope.get("mscale_all_dim", 0)
    factor = rope["factor"]
    if not mscale_all_dim or factor <= 1:
        return
    m = 0.1 * mscale_all_dim * math.log(factor) + 1.0
    for layer in model.model.layers:
        layer.self_attn.scaling *= m * m
    print(f"applied yarn mscale^2 softmax-scale correction ({m * m:.6f})")


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

    decode_output = None
    max_new_tokens = 0
    if "--decode-output" in rest:
        at = rest.index("--decode-output")
        if at + 1 >= len(rest):
            print("--decode-output needs a path", file=sys.stderr)
            return 2
        decode_output = rest[at + 1]
        del rest[at : at + 2]
    if "--max-new-tokens" in rest:
        at = rest.index("--max-new-tokens")
        if at + 1 >= len(rest):
            print("--max-new-tokens needs a positive integer", file=sys.stderr)
            return 2
        max_new_tokens = int(rest[at + 1])
        del rest[at : at + 2]
    if decode_output is not None and max_new_tokens <= 0:
        print("--decode-output requires --max-new-tokens > 0", file=sys.stderr)
        return 2

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
    fix_yarn_softmax_scale(model)

    with torch.no_grad():
        input_ids = torch.tensor([ids], dtype=torch.long, device=device)

        prompt_ids = list(ids)
        generated = None
        if decode_output is not None:
            sequence = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
                eos_token_id=None,
                pad_token_id=tokenizer.eos_token_id,
            )[0].cpu().tolist()
            generated = sequence[len(ids) :]
            if len(generated) != max_new_tokens:
                raise RuntimeError(
                    f"requested {max_new_tokens} decode tokens but HF emitted "
                    f"{len(generated)}"
                )
            # The IXRL container then covers the whole teacher-forced
            # sequence, not just the prompt: the cached-decode test needs
            # HF's margins at every generated position to tell a bf16
            # near-tie flip from a real cache defect.
            ids = ids + generated
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

    if decode_output is not None:
        # IXDG is deliberately separate from IXRL: the latter is a dense
        # teacher-forced logit oracle, while this is the exact sequence emitted
        # by HF's cached greedy generation loop. The C++ test feeds every
        # winning token back through InferX's scheduler and paged KV cache.
        with open(decode_output, "wb") as f:
            f.write(b"IXDG")
            f.write(struct.pack("<III", 1, len(prompt_ids), len(generated)))
            f.write(struct.pack(f"<{len(prompt_ids)}i", *prompt_ids))
            f.write(struct.pack(f"<{len(generated)}i", *generated))
        print(
            f"wrote {len(generated)} cached greedy decode tokens to "
            f"{decode_output}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
