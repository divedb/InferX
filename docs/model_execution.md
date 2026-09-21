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
LlamaModel / Qwen3Model / ...
    |
    v
ops
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
  kv_cache, ctx)` returns `[num_seqs, vocab]` logits for the requested rows.
  `Model::Load` selects the implementation from the checkpoint's
  `model_type`; new architectures register there and nowhere else.
- **Architecture directories** own their weights and forward flows:
  `models/llama/` and `models/qwen3/` each define their layer weights,
  checkpoint layout, and the sequence of ops their forward runs. Shared
  checkpoint machinery (config parsing, safetensors, bf16 weight upload)
  lives at `models/` top level and in `src/models/weight_upload.*`.
- **ops** holds the reusable, backend-portable operations the forwards call.
  Each op is a free function: a public header with the agnostic API and
  config, one common source that validates arguments and dispatches on the
  execution context's device, and one directory per backend
  (`ops/cuda/`, `ops/cpu/`) exposing namespaced implementations
  (`ops::cuda::RmsNorm`, `ops::cpu::RmsNorm`). Backend details — FlashInfer
  kernels and dtype support on CUDA, Highway vectorization and multi-target
  dispatch on CPU — stay inside their backend directories.

The architecture forwards are filled in as ops land; today `RmsNorm` exists
(writing into an explicit output tensor, with the Gemma `1 + weight` variant
configurable) and the Llama/Qwen3 forwards validate their inputs and report
the ops they await. Weight loading is complete for both architectures (the
Qwen3 path is covered by `tests/model_test.cc` against a real 0.6B
checkpoint), and `tests/model_runner_test.cc` pins the runner contract with a
fake model: the batch the model receives, the tokens fed back after sampling,
and that failures surface before Forward runs.
