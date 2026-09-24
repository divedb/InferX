# Model execution

```
Scheduler
    |
    v
SchedulerOutput
    |
    v
ModelRunner::Run(...)
    |  prepare/extract execution parameters
    v
Model::Forward(...)
    |
    v
causal::CausalLM (shared dense decoder stack)
    |
    v
components (attention, mlp, qkv_linear, ...)  --  ops
```

## Responsibilities

- **Scheduler** decides which requests/sequences execute and owns KV block
  allocation.
- **ModelRunner** (`models/model_runner.{h,cc}`) is the execution entry point
  between the scheduler and the model. It converts a `SchedulerOutput` into a
  flat `ModelInput` — token ids, positions, attention metadata (indptrs, block
  tables, last-page lengths), and the per-sequence logit rows — hands it to
  `Model::Forward`, greedily samples the returned logits, and tracks the
  per-request state that survives between steps (block tables, computed-token
  watermarks, the last sampled token). It owns the execution lane (device
  runtime + stream) and the paged KV pool. It contains no model-specific
  computation.
- **`Model`** (`models/model.h`) is the common interface: `Forward(input,
  state, ctx)` returns `[num_seqs, vocab]` logits for the requested rows.
  `Model::Load` resolves the checkpoint's `model_type` / `architectures`
  through the family registry.
- **Family registry** (`models/model_registry.{h,cc}`) maps checkpoint
  identity to an implementation at the lowest of three escalating tiers:
  *tier 1* — a `Family` row of knobs (`qk_norm`, `plus_one_norm`, checkpoint
  name layout) describing a standard dense causal decoder; llama is one such
  row. *tier 2* — a `translate` function producing a `DecoderConfig` for
  families whose configs differ beyond knobs (MoE fields, layer types;
  `models/qwen3.cc`, one flat file registering three identities). *tier 3* —
  a `build` function owning everything, for architectures no `DecoderConfig`
  can express (encoder-decoder, multimodal towers, native SSM). Tier-2/3
  families self-register from their own translation units into the
  `inferx_model_families` archive (whole-archived; see the CONSTRAINT in
  `src/CMakeLists.txt`); adding one never edits `model_registry.cc`. The
  composition — parse, translate, validate, shard, `BuildCausalLM` — lives
  once in `BuildFamily`, so no family repeats it.
- **`causal/`** is the generic dense decoder: `DecoderStack` (embedding ->
  pre-norm layers -> final norm) plus `CausalLM`, the served `Model`, and the
  weight mapping that loads any Llama-style checkpoint layout into it.
  `BuildCausalLM` validates the total config, rewrites each attention block's
  head counts rank-local under the `ParallelConfig` seam
  (`engine/parallel_config.h`, vLLM ParallelConfig analogue; defaults shard
  nothing), and loads weights with this rank's row slices.
- **`components/`** holds the reusable transformer pieces (attention, mlp,
  moe, norm, rope, linear, qkv_linear): config + weights + execution as
  concrete, non-virtual pieces the stack composes. `qkv_linear` is the
  QKVParallelLinear analogue: `ShardQkv` derives one rank's head geometry
  (query heads divided across ranks, KV heads divided or replicated below the
  rank count), and attention always projects through one fused
  `[query | key | value]` GEMM whose per-projection weights are views into
  the packed allocation.
- **ops** holds the reusable, backend-portable operations the forwards call.
  Each op is a free function: a public header with the agnostic API and
  config, one common source that validates arguments and dispatches on the
  execution context's device, and one directory per backend
  (`ops/cuda/`, `ops/cpu/`) exposing namespaced implementations
  (`ops::cuda::RmsNorm`, `ops::cpu::RmsNorm`). Backend details — FlashInfer
  kernels and dtype support on CUDA, Highway vectorization and multi-target
  dispatch on CPU — stay inside their backend directories.

Weight loading is covered end to end: `tests/model_test.cc` runs the real
Qwen3-0.6B checkpoint through forward, `tests/unit/models/qkv_linear_test.cc`
pins the shard math and the packed QKV row layout (block order, view
aliasing, rank slices, KV replication) against synthetic safetensors, and
`tests/unit/models/family_config_test.cc` covers tier translation plus the
registry canary that fails loudly if a family's registration ever drops out
of the link. `tests/model_runner_test.cc` pins the runner contract with a
fake model: the batch the model receives, the tokens fed back after sampling,
and that failures surface before Forward runs.
