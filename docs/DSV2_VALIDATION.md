# DeepSeek-V2-Lite validation on a rented GPU

The §18.7 D5 acceptance procedure. DeepSeek-V2-Lite is ~31.4 GB of bf16
weights, which the 16 GB development GPU cannot hold, so first serving and the
golden-logits gate run on a rented box (§18.6, option "rented GPU at bf16" —
the TP_VALIDATION.md pattern). Everything below is scripted or env-gated;
nothing needs code changes on the rented machine except the one decision the
session exists to make.

## What this session decided (settled 2026-08-09)

1. **The RoPE convention** — settled: **de-interleaved**. Not by argmax
   voting in the end, but directly: with the load-time row gather, InferX's
   post-rope q/k tensors match HF's `apply_rotary_emb` outputs to bf16
   rounding, and the YaRN inv_freq tables agree to 1e-8. The gather is
   hardcoded in `src/model/deepseek_v2.cc` (`DeinterleaveMap`) and the
   `INFERX_DSV2_ROPE_DEINTERLEAVE` toggle is deleted.
2. **Whether the whole composition matches HF** — it does, with one landmine
   **in HF, not here**: transformers' integrated `DeepseekV2Attention`
   (measured on 5.14.1) uses a bare `qk_head_dim**-0.5` softmax scale,
   dropping the yarn `mscale**2` factor (~1.5896) that the checkpoint's own
   `modeling_deepseek.py` (and vLLM) apply and that InferX applies. Left
   uncorrected it moved 17 of 65 teacher-forced argmaxes; every layer's
   hidden state past the first attention diverged while position 0 — whose
   1-element softmax is scale-invariant — stayed clean, which is the
   signature to remember. `scripts/gen_deepseek_logits.py` patches the scale
   back before writing goldens; references generated without it are wrong.
3. **First serving numbers** for the unabsorbed MLA + per-expert-GEMM path,
   which set the baseline the D6 performance items are measured against
   (still pending).

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

# 3. Golden logits (transformers + torch in a venv). The script loads the
#    integrated transformers implementation (trust_remote_code=False) and then
#    restores the yarn mscale^2 softmax scale the integrated port drops --
#    do not generate references any other way (see "What this session
#    decided" #2).
python -m venv .venv-ref && . .venv-ref/bin/activate
pip install torch transformers accelerate
scripts/gen_deepseek_logits.py /ckpt/dsv2-lite testdata/deepseek_v2_lite_logits.bin
# Also pin multi-step greedy generation through HF's real use_cache path. Use
# the Chat checkpoint here when validating the checkpoint served in production.
# With --decode-output, the IXRL logits file spans the whole teacher-forced
# prompt+continuation, giving the cached-decode gate HF's margins at every
# generated position.
scripts/gen_deepseek_logits.py /ckpt/dsv2-lite-chat \
    testdata/deepseek_v2_lite_chat_logits.bin \
    --decode-output testdata/deepseek_v2_lite_decode.bin \
    --max-new-tokens 128 --prompt "<a long, representative prompt>"
# With >= 70 GB RAM, prefer the stricter fp32 CPU reference (hours, one-off):
#   scripts/gen_deepseek_logits.py /ckpt/dsv2-lite out.bin --fp32

# 4. The gate.
export INFERX_TEST_DEEPSEEK_CHECKPOINT=/ckpt/dsv2-lite
cd build && ctest -R DeepseekV2Reference --output-on-failure

# For the cached-decode gate, point both inputs at artifacts made from the same
# Chat checkpoint. The test uses Scheduler::PrepareStep/CommitStep and the real
# paged latent cache, and compares every generated token ID rather than text.
export INFERX_TEST_DEEPSEEK_CHECKPOINT=/ckpt/dsv2-lite-chat
export INFERX_TEST_DEEPSEEK_LOGITS=../testdata/deepseek_v2_lite_chat_logits.bin
export INFERX_TEST_DEEPSEEK_DECODE=../testdata/deepseek_v2_lite_decode.bin
ctest -R DeepseekV2Reference --output-on-failure

# 5. Serve, and check the template end to end.
./build/src/server/inferx-serve --model /ckpt/dsv2-lite-chat \
    --kv-blocks 8192 --block-size 16 --no-cuda-graphs
curl -s localhost:8080/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"dsv2-lite-chat","messages":[{"role":"user","content":"What is 2+2?"}],"max_tokens":32}'

# 6. Baseline against vLLM on the same box (bench-results/ naming convention:
#    deepseek-v2-lite-<engine>-<date>).
MODEL=/ckpt/dsv2-lite-chat MODEL_NAME=deepseek-v2-lite ./scripts/run_vllm.sh
bench/serve_bench.py ...
```

## Interpreting the gate

- Both gates tolerate an argmax flip only when the reference's own top-2
  margin is within `kTieTolerance` (0.25, two bf16 ULPs at logit magnitude
  ~32): the chat prompt has *exact* bf16 ties at several positions, where no
  winner exists to agree with. The cached-decode gate teacher-forces the
  reference token after a tolerated tie so one flip doesn't invalidate the
  rest of the trajectory. 2026-08-09 measurement: 9 tie flips over 193
  teacher-forced positions, 2 over 128 cached decode steps (one of which HF's
  own generate loop also disagrees with its own full forward on), zero hard
  mismatches on either checkpoint or on the 157-token long prompt.
- **Argmax differs at many positions by wide margins**: suspect, in order:
  the reference itself (generated without the mscale^2 patch — see "What
  this session decided" #2), the loader mapping (rerun with `GetChecked`
  errors in the log), then the YaRN frequency table.
- **Divergence that spares position 0 but hits everything else, growing with
  depth**: an attention-score-scale or rope-angle class of bug — position
  0's softmax is over one element, so scale and rotation cannot touch it.
  Compare `MlaAttentionLayer::softmax_scale()` ≈ 0.114721 against
  `q_head_dim**-0.5 * yarn_get_mscale(40, 0.707)**2` first.
- **A single position differs beyond tolerance**: check the router's 6th/7th
  margin at that token before concluding anything — bf16 routing
  discontinuity, the gpt-oss precedent (see the note in
  `deepseek_v2_reference_test.cc`).
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
