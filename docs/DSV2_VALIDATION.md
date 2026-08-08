# DeepSeek-V2-Lite validation on a rented GPU

The §18.7 D5 acceptance procedure. DeepSeek-V2-Lite is ~31.4 GB of bf16
weights, which the 16 GB development GPU cannot hold, so first serving and the
golden-logits gate run on a rented box (§18.6, option "rented GPU at bf16" —
the TP_VALIDATION.md pattern). Everything below is scripted or env-gated;
nothing needs code changes on the rented machine except the one decision the
session exists to make.

## What this session decides

1. **The RoPE convention** — the standing ARCHITECTURE.md caveat. Our MLA rope
   rotates half-split pairs; HF's DeepSeek code de-interleaves storage pairs
   first. The two are settled by measurement, not argument:
   `INFERX_DSV2_ROPE_DEINTERLEAVE=1` switches the loader to the de-interleaved
   reading, and whichever setting passes the argmax gate is the convention.
   **After this session,** hardcode the winner in `src/model/deepseek_v2.cc`
   (see `RopeDeinterleaveRequested`) and delete the env var: a semantics
   toggle must not survive validation.
2. **Whether the whole composition matches HF** — YaRN table, mscale softmax
   scale, dense/MoE schedule, routing, ungated shared experts, loader mapping.
3. **First serving numbers** for the unabsorbed MLA + per-expert-GEMM path,
   which set the baseline the D6 performance items are measured against.

## Box requirements

| Requirement | Why |
|---|---|
| GPU ≥ 48 GB (L40S, A6000 Ada, A100) | 31.4 GB weights + latent KV pool + activations, sm_89+ preferred to match `INFERX_CUDA_ARCHS` (set `-DINFERX_CUDA_ARCHS` for other archs) |
| CUDA toolkit 13.0+, driver r580+ | configure-time floor, as on the dev box |
| ~40 GB disk | checkpoint + build tree |
| ~70 GB host RAM (optional) | only for `--fp32` reference logits, which make every disagreement ours; bf16 GPU goldens are the default |

## Procedure

```bash
# 1. Clone, build (defaults are the CUDA build).
git clone <repo> inferx && cd inferx
./scripts/bootstrap.sh

# 2. Checkpoint. Base model for logits; -Chat for template/serving checks.
pip install -U "huggingface_hub[cli]"
huggingface-cli download deepseek-ai/DeepSeek-V2-Lite --local-dir /ckpt/dsv2-lite
huggingface-cli download deepseek-ai/DeepSeek-V2-Lite-Chat --local-dir /ckpt/dsv2-lite-chat

# 3. Golden logits (transformers + torch in a venv; trust_remote_code is
#    required -- the checkpoint ships its own modeling code).
python -m venv .venv-ref && . .venv-ref/bin/activate
pip install torch transformers accelerate
scripts/gen_deepseek_logits.py /ckpt/dsv2-lite testdata/deepseek_v2_lite_logits.bin
# With >= 70 GB RAM, prefer the stricter fp32 CPU reference (hours, one-off):
#   scripts/gen_deepseek_logits.py /ckpt/dsv2-lite out.bin --fp32

# 4. The gate.
export INFERX_TEST_DEEPSEEK_CHECKPOINT=/ckpt/dsv2-lite
cd build && ctest -R DeepseekV2Reference --output-on-failure

# 5. If (and only if) the argmax gate fails at essentially every position:
INFERX_DSV2_ROPE_DEINTERLEAVE=1 ctest -R DeepseekV2Reference --output-on-failure
# Record which setting passed. That is the RoPE convention. Hardcode it.

# 6. Serve, and check the template end to end.
./build/src/server/inferx-serve --model /ckpt/dsv2-lite-chat \
    --kv-blocks 8192 --block-size 16 --no-cuda-graphs
curl -s localhost:8080/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"dsv2-lite-chat","messages":[{"role":"user","content":"What is 2+2?"}],"max_tokens":32}'

# 7. Baseline against vLLM on the same box (bench-results/ naming convention:
#    deepseek-v2-lite-<engine>-<date>).
MODEL=/ckpt/dsv2-lite-chat MODEL_NAME=deepseek-v2-lite ./scripts/run_vllm.sh
bench/serve_bench.py ...
```

## Interpreting the gate

- **Argmax equal at every position, both toggle settings tried once**: done.
  Record the passing convention, the worst-deviation number, and the serving
  throughput in ARCHITECTURE.md's measurement section.
- **Argmax differs at every position under both settings**: the problem is not
  the interleave. Suspect, in order: the mscale softmax scale (compare
  `MlaAttentionLayer::softmax_scale()` ≈ 0.114721 against
  `q_head_dim**-0.5 * yarn_get_mscale(40, 0.707)**2` in the HF code), the
  YaRN frequency table, then the loader mapping (rerun with `GetChecked`
  errors in the log).
- **A single position differs**: check the router's 6th/7th margin at that
  token before concluding anything — bf16 routing discontinuity, the gpt-oss
  precedent (see the note in `deepseek_v2_reference_test.cc`).
- The 5% span-normalized magnitude bound is diagnostic, not the gate, for the
  same reason as gpt-oss: a bf16 reference is not more precise than what it
  checks.

## Non-goals of this session

Long-context (>4096) YaRN behavior — the goldens are short prompts; the YaRN
frequencies and mscale are exercised at every position, but serving claims
beyond the original window need a separate long-context perplexity check
before the registry advertises more (§18.5). Performance work — D6 starts
from the numbers this session records, on whichever hardware serving lands.

Expected serving shape, for calibration rather than surprise: decode is
O(context) per step per layer (unabsorbed reconstruction) and each MoE layer
runs 64 cuBLASLt calls behind a host sync, so tokens/s will be far below the
Qwen2 path. That is the documented D6 gap, not a defect of this session.
