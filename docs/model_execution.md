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
causal::CausalLM<Traits> (typed dense decoder stack)
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
- **Family registry** (`models/model_registry.{h,cc}`) maps checkpoint identity
  to factories in `src/models/families/`. Each dense family declares its
  architecture, checkpoint names/layout, normalization placement/type, attention
  type, and MLP type in a traits struct. `causal::MakeFamily<Traits>` generates
  the factory and its configuration translator. Families with additional config
  fields supply a translator; structurally different models can supply a custom
  `Family::build` function returning `Model`. Built-in factory references are
  explicit, so static linking needs neither registration initializers nor a
  whole-archive family library. Conflicting known model type/architecture
  identities are rejected.
- **`causal/`** contains `CausalLM<Traits>`, `DecoderStack<Traits>`, and
  `DecoderLayer<Traits>` as private implementation templates. The public
  `causal/decoder_config.h` holds runtime dimensions, per-layer configuration,
  and the loader's weight bundle. `PrepareCausalLM` shares validation, sharding,
  loading, and head weight tying. It rejects unsupported execution before
  reading tensors. `DecoderWorkspace` shares input validation, buffer allocation,
  and attention planning across template instantiations.
- **Ownership and execution:** each concrete decoder layer owns its norms,
  attention, MLP, and their weights. The stack owns the embedding, final norm,
  and reusable activation workspace. The runner supplies persistent layer state.
  The layer retains the fused residual/RMSNorm schedule: the MLP residual is
  added during the following layer's input norm or the stack's final norm.
  `Model::Forward` remains the serving boundary, including its borrowed-output
  lifetime and CUDA graph contract.
- **`loading/`** separates checkpoint names (`WeightNames`) from source storage
  (`WeightLayout`). Paths omit `.weight`/`.bias`; layer-local names are relative
  to `layers.<index>.`. Separate Q/K/V tensors and fused `[Q | K | V]` row-major
  `[out, in]` tensors both load into the same packed rank-local representation.
  Gate/up names identify semantic roles independently of checkpoint spelling.
  Loading checks full shapes before slicing, replicates KV heads when needed,
  and preserves projection views into packed allocations.
- **`components/`** contains concrete `GqaAttention`, `GatedMlp`, and `RmsNorm`
  components over shared execution functions. Traits select structural choices;
  dimensions, epsilon, RoPE parameters, and weight tying remain runtime data.
  QKV and output bias are represented separately. Backend kernels remain in ops.
  Tensor-parallel geometry and loading are supported, but execution with more
  than one rank remains rejected until collectives are implemented.

- **ops** holds the reusable, backend-portable operations the forwards call.
  Each op is a free function: a public header with the agnostic API and
  config, one common source that validates arguments and dispatches on the
  execution context's device, and one directory per backend
  (`ops/cuda/`, `ops/cpu/`) exposing namespaced implementations
  (`ops::cuda::RmsNorm`, `ops::cpu::RmsNorm`). Backend details — FlashInfer
  kernels and dtype support on CUDA, Highway vectorization and multi-target
  dispatch on CPU — stay inside their backend directories.


The executable families are Llama 3.x, Qwen2.5, Qwen3 (dense, MoE, and Next),
Mistral, Gemma 3, Mixtral, DeepSeek-V3, and gpt-oss. Each is a thin identity
(traits plus a config translator over the shared `AttentionDecoderConfig`);
every capability they use is implemented once as a shared component:

- **Attention** — GQA with optional Q/K norms (plain or plus-one), biased
  projections, per-layer sliding windows, attention sinks, scale overrides,
  gated outputs, and RoPE scaling (linear, llama3, yarn with its softmax
  temperature). FlashInfer runs the fast path; a generic kernel covers sinks
  and geometries outside its template set.
- **Feed-forward** — one gated-MLP execution with runtime activation flavor
  (silu, Gemma's gelu-tanh, gpt-oss's clamped SwiGLU-oai) and optional packed
  biases; MoE layers route through one kernel covering renormalized softmax
  (Mixtral, Qwen3, gpt-oss) and DeepSeek's sigmoid/grouped scheme, dispatch
  experts through the same gated-MLP ops, and reuse the dense path for shared
  experts.
- **MLA** — DeepSeek's latent attention runs decompressed through the same
  paged pool and kernels as GQA; the compressed latent cache remains a future
  `KvLayout` mode.
- **Recurrent layers** — Qwen3-Next's Gated DeltaNet executes the reference
  recurrence sequentially with per-slot state in a `RecurrentStatePool`; the
  KV pool carries one layout per layer so hybrid stacks allocate both.
- **Residual styles** — pre-norm (most families) and Gemma's
  output-normalized sandwich share the same fused AddForward seam.
- **Quantized checkpoints** — gpt-oss's MXFP4 experts dequantize at load
  (E8M0 block scales, interleaved gate/up rows and biases); gpt-oss therefore
  needs device memory for its dequantized experts. GPTQ/AWQ remain unsupported.

To add a family, declare traits in `src/models/families/<name>.cc`, return
`MakeFamily<Traits>()` (or supply a config translator), and add its factory to
`families.h`, the registry, and `src/CMakeLists.txt`. Components never inspect
architecture strings or checkpoint tensor names.


Weight loading is covered end to end: `tests/model_test.cc` runs the real
Qwen3-0.6B checkpoint through forward, `tests/unit/models/parallel_linear_test.cc`
pins the shard math and the packed QKV row layout (block order, view
aliasing, rank slices, KV replication) against synthetic safetensors, and
`tests/unit/models/family_config_test.cc` covers traits-driven translation,
architecture aliases, malformed configuration, and early rejection of unsupported
execution. Loader tests also cover fused QKV source rows and custom gate/up name
mappings. `tests/model_runner_test.cc` pins the runner contract with a fake model: the batch the model receives, the tokens fed back after sampling,
and that failures surface before Forward runs.
