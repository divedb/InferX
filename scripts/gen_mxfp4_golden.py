#!/usr/bin/env python3
"""Pin the MXFP4 bit layout, so the CUDA decoder is checked against a fact.

MXFP4 is the format gpt-oss-20b's expert weights are stored in and the one thing
on the critical path that cannot be worked around: dequantizing them costs 38 GB
where the packed form costs 10. Getting the layout wrong produces weights that
are wrong *and still look like a language model*, which is the most expensive
kind of wrong. So it gets settled once, here, against HuggingFace's own decoder,
and the answer is written to a file the C++ test reads.

The format, as verified against `transformers.integrations.mxfp4`:

  * 32 values share one block. `blocks` is uint8 `[..., G, 16]` — 16 bytes,
    two values per byte — and `scales` is uint8 `[..., G]`, one per block.
  * A scale is **E8M0**: a bare exponent, bias 127, so the block multiplier is
    `2^(scale - 127)`. No mantissa, which makes dequantization a shift rather
    than a multiply.
  * A value is **FP4 E2M1**, looked up in
    `[0, .5, 1, 1.5, 2, 3, 4, 6, -0, -.5, -1, -1.5, -2, -3, -4, -6]`.
  * **The low nibble comes first.** Byte `i` holds element `2i` in its low four
    bits and element `2i+1` in its high four. This is the detail that has no
    right answer from first principles and is why this script exists.

One further finding this script pins, because it is worth more than it looks.
HF's `convert_moe_packed_tensors` returns `[E, hidden, 2·inter]` — its module
computes `x @ W` — while the checkpoint's own axis order is
`[E, 2·inter, hidden]`. **That raw order is exactly `MoeWeights::gate_up`**,
because `LinearBF16` computes `y = x·Wᵀ`. So the loader needs no transpose; the
`.T` below is undoing HF's convention, not ours.

Output:

    magic    'IXM4'         4 bytes
    version  u32 = 1
    cases    u32
    per case:
      label_len u32, label bytes (no NUL)
      rows      u32          rows of the weight matrix in this case
      blocks    u32          blocks per row (G)
      blocks    u8 * rows * G * 16
      scales    u8 * rows * G
      expected  f32 * rows * G * 32   (row-major, our axis order)

Usage:
    scripts/gen_mxfp4_golden.py <checkpoint-dir> <output.bin>
"""

import json
import struct
import sys

MAGIC = b"IXM4"
VERSION = 1
ROWS_PER_CASE = 32

FP4_VALUES = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
              -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]


def decode_mxfp4(blocks, scales, np):
    """The format, written out. `blocks` [..., G, 16] u8, `scales` [..., G] u8."""
    lut = np.array(FP4_VALUES, dtype=np.float32)

    out = np.empty(blocks.shape[:-1] + (blocks.shape[-1] * 2,), dtype=np.float32)
    out[..., 0::2] = lut[blocks & 0x0F]   # low nibble first
    out[..., 1::2] = lut[blocks >> 4]

    exp = scales.astype(np.int32) - 127   # E8M0, bias 127
    out *= np.ldexp(np.float32(1.0), exp)[..., None]

    return out.reshape(blocks.shape[:-2] + (-1,))


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__)
        return 2

    import numpy as np
    import torch
    from safetensors import safe_open
    from transformers.integrations.mxfp4 import convert_moe_packed_tensors

    ckpt_dir = sys.argv[1].rstrip("/")
    out_path = sys.argv[2]

    index = json.load(open(f"{ckpt_dir}/model.safetensors.index.json"))
    weight_map = index["weight_map"]

    # Both ends of the checkpoint and both projections: an indexing bug that
    # only bites on the last layer or on `down_proj` is exactly the kind that
    # survives a one-case test.
    cases = [
        ("layer0.gate_up.expert0", "model.layers.0.mlp.experts.gate_up_proj", 0),
        ("layer0.down.expert0", "model.layers.0.mlp.experts.down_proj", 0),
        ("layer23.gate_up.expert31", "model.layers.23.mlp.experts.gate_up_proj", 31),
    ]

    records = []

    for label, base, expert in cases:
        blocks_name = f"{base}_blocks"
        scales_name = f"{base}_scales"

        with safe_open(f"{ckpt_dir}/{weight_map[blocks_name]}", framework="np") as f:
            blocks = f.get_slice(blocks_name)[expert:expert + 1][0, :ROWS_PER_CASE]
        with safe_open(f"{ckpt_dir}/{weight_map[scales_name]}", framework="np") as f:
            scales = f.get_slice(scales_name)[expert:expert + 1][0, :ROWS_PER_CASE]

        mine = decode_mxfp4(blocks, scales, np)

        # The cross-check. HF returns the transpose of our axis order, so this
        # compares the same numbers through two independent decoders and two
        # conventions -- and it has to be *exact*, not close: both are looking
        # up the same 16 representable values and scaling by a power of two, so
        # there is no rounding for a tolerance to absorb.
        theirs = convert_moe_packed_tensors(
            torch.from_numpy(blocks[None]), torch.from_numpy(scales[None]),
            dtype=torch.float32).numpy()
        theirs = np.ascontiguousarray(theirs[0].T)

        if not np.array_equal(mine, theirs):
            worst = float(np.abs(mine - theirs).max())
            print(f"error: {label} disagrees with transformers (max {worst})",
                  file=sys.stderr)
            return 1

        print(f"{label}: {mine.shape} exact against transformers, "
              f"{float((mine != 0).mean()):.1%} nonzero")

        records.append((label, blocks, scales, mine))

    with open(out_path, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<II", VERSION, len(records)))

        for label, blocks, scales, expected in records:
            raw = label.encode()
            f.write(struct.pack("<I", len(raw)))
            f.write(raw)
            f.write(struct.pack("<II", blocks.shape[0], blocks.shape[1]))
            f.write(blocks.astype("<u1").tobytes())
            f.write(scales.astype("<u1").tobytes())
            f.write(expected.astype("<f4").tobytes())

    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
