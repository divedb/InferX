# InferX Model Architecture — Analysis and Reimplementation Proposal

Status: design only; no code changed. Scope: the model layer under
`include/inferx/models/` and `src/models/`, and the `Model`/`ModelRunner`
boundary.

This document studies how vLLM and SGLang structure model implementations, then
proposes a clean, modular, extensible reimplementation of the InferX model layer.
It is deliberately concrete about interfaces and boundaries so implementation can
begin without repeating the research.

---

## 1. Reference findings

### 1.1 vLLM (`/home/gc/vllm`)

**Registry.** `vllm/model_executor/models/registry.py:45-324` maps HF
`architectures[0]` strings to `(module, class)` pairs in flat dicts
(`_TEXT_GENERATION_MODELS`, `_MULTIMODAL_MODELS`, `_SPECULATIVE_DECODING_MODELS`,
…), merged into `_VLLM_MODELS` and wrapped as lazy entries
(`_LazyRegisteredModel`, `registry.py:425-522`). Out-of-tree models register via
`ModelRegistry.register_model(arch, cls_or_"module:Class")`
(`registry.py:561-601`). Resolution order is `model_impl` → transformers fallback
→ built-in (`resolve_model_cls`, `registry.py:778-829`). Capabilities are computed
as a `_ModelInfo` dataclass of boolean flags (`registry.py:347-388`) derived from
structural protocols.

**Model contract.** Split across two files by design (`interfaces_base.py:28-30`):

- `VllmModel`: `__init__(*, vllm_config, prefix="")`; `forward(input_ids, positions)`.
- `VllmModelForTextGeneration`: adds `compute_logits(hidden_states) -> Optional[Tensor]`.
- Optional capability protocols in `interfaces.py`: `SupportsMultiModal`,
  `SupportsLoRA`, `SupportsPP` (`make_empty_intermediate_tensors`), `HasInnerState`,
  `IsHybrid`, `MixtureOfExperts` (expert counts + `set_eplb_state`), etc. — detected
  structurally, not by inheritance.
- `load_weights(weights) -> set[str]` is the explicit loader contract; strict
  completeness is enforced for non-quantized models
  (`model_loader/default_loader.py:263-277`).

**Attention is hidden behind an op.** Models construct
`Attention(num_heads, head_size, ..., prefix=...)` (`attention/layer.py:74-104`),
which selects a backend by `(head_size, dtype, kv_cache_dtype, block_size, use_mla,
has_sink, use_sparse)` and registers itself into a static forward context keyed by
`prefix`. `forward(query, key, value)` reads `attn_metadata` from the context;
models never pass attention metadata. MLA models build an `MLAModules` +
`MultiHeadLatentAttention` wrapper (`deepseek_v2.py:830-984`), and MTP reuses the
target's `DecoderLayer` as a child module (`deepseek_mtp.py:68-69`).

**Composition & naming.** Everything is an `nn.Module`;
`make_layers(num_hidden_layers, layer_fn, prefix)` builds the stack
(`models/utils.py:615-633`), `maybe_prefix`/`extract_layer_index` thread parameter
names and per-layer indices (`utils.py:684-723`). `init_vllm_registered_model`
instantiates an inner registered model for composite architectures
(`utils.py:298-319`).

**Weight loading.** Per-parameter `weight_loader` callables (`parameter.py`:
`BasevLLMParameter`, `_ColumnvLLMParameter.load_merged_column_weight`/`load_qkv_weight`,
`RowvLLMParameter`) plus a recursive `AutoWeightsLoader` (`models/utils.py:84-295`)
that prefers each child's own `load_weights` and otherwise calls
`param.weight_loader`/`default_weight_loader`. `WeightsMapper` (`utils.py:32-81`)
declares substring/prefix/suffix renames with `None` = drop. Fused checkpoint names
are absorbed by `stacked_params_mapping` (`llama.py:419-483`) and
`packed_modules_mapping`.

**Parallel layers.** `LinearBase`/`ReplicatedLinear`/`ColumnParallelLinear`/
`MergedColumnParallelLinear`/`QKVParallelLinear`/`RowParallelLinear`
(`layers/linear.py`) each own their sharding math as a function of
`(tp_size, total_heads, output_sizes)`; the parameter object holds
`output_dim`/`input_dim`/`packed_dim` so shard loading is data-driven. `forward`
returns `(output, bias_or_None)`.

**Key extensibility seams:** (a) arch string → lazy class; (b) capability protocols,
not inheritance; (c) per-parameter weight loaders + recursive `AutoWeightsLoader`;
(d) `prefix`-keyed parameter tree; (e) attention/KV hidden behind a backend-selected
op; (f) `init_vllm_registered_model` for composition; (g) `VerifyAndUpdateConfig`
overrides in `models/config.py`.

### 1.2 SGLang / mini-sglang

**mini-sglang** (`/home/gc/mini-sglang/python/minisgl`):

- `models/register.py:5-21`: explicit `arch → (module, class)` dict, lazily
  imported and constructed with the config.
- `models/base.py:12-14`: the entire interface is
  `class BaseLLMModel(ABC): def forward(self) -> Tensor`. Everything else (weights,
  sharing) comes from `BaseOP` (`layers/base.py:15-53`), which provides
  reflection-based `state_dict`/`load_state_dict` over `self.__dict__`, keyed by
  dotted paths; `OPList` namespaces children by index. **Names are the weight-loading
  interface** — there is no remap layer; the streaming loader `models/weight.py`
  emits runtime-schema names (split/merge/stack q/k/v and gate/up, stack experts).
- Runtime metadata reaches deep layers through a process-global `Context`/`Batch`
  (`core.py:100-136`): `input_ids`, `positions`, `out_loc` (KV write locations),
  `attn_metadata`, `page_table`, `attn_backend`, `kv_cache`.
- Models are ~80 lines: `Qwen3ForCausalLM` wires `Qwen3Model` + `ParallelLMHead`;
  `Qwen3Model` is embedding + `OPList` + final norm; the layer differs only by flags
  passed to the shared `RopeAttn`/`GatedMLP` in `models/utils.py`.
- Backends are behavior-only ABCs (`attention/base.py`:
  `BaseAttnBackend.forward/prepare_metadata/init_capture_graph/...`; `moe/base.py`;
  `kvcache/base.py`) selected through a `Registry`.

**Full SGLang** (installed 0.5.19): the scale patterns are (i) `EntryClass`
self-registration + `pkgutil` auto-scan (`srt/models/registry.py:94-131`),
(ii) explicit `ForwardBatch` passed into
`forward(input_ids, positions, forward_batch, ...)` rather than globals
(`forward_batch_info.py:394`), (iii) per-layer class chosen from
`config.layers_block_type` for hybrid attention (`qwen3_next.py:901-918`),
(iv) `RadixAttention` wrapper hiding the backend (`radix_attention.py:98`),
(v) a dedicated `LogitsProcessor` separating sampling from the model, and
(vi) parameter-level `weight_loader`s + `packed_modules_mapping`.

### 1.3 Convergent principles to adopt

1. **Model = composition of reusable modules.** Architecture code is wiring + name
   mapping, not execution.
2. **Names are the loading interface**, with a declarative remap and per-parameter
   shard rules.
3. **Cross-cutting per-step state is an explicit context object**, not globals and
   not a 15-argument function.
4. **Cache/recurrent state is declared as specs and acquired as opaque handles**, so
   hybrid caches are first-class.
5. **Capabilities are a descriptor**, consumed by the runner for cache sizing and
   graph policy.
6. **Backends (attention/MoE/quant) live behind registries**, selected by
   capability, never referenced by model code.
7. **Open extension**: out-of-tree registration for architectures and components.

---

## 2. Current InferX model layer

### 2.1 What exists

- Facade: `Model` with `Load`, `config()`, `StateRequirements()`,
  `Forward(ModelInput, ModelState, OpContext&, CommBackend&)`
  (`model.h:60-98`).
- Registry: `Family{model_type, architecture, build, translate}` + explicit
  `std::array` in `model_registry.cc:11-18`; resolved by `ResolveFamily`.
- Typed dense path: `ModelTraits` concept (`model_traits.h:17-27`) →
  `MakeFamily<Traits, Translate>` (`causal/build.h:29-54`) → `CausalLM<Traits>` →
  `DecoderStack<Traits>` → `DecoderLayer<Traits>`.
- Components (Config/Weights/Run triad): `AttentionConfig/Weights/RunAttention` +
  `GqaAttention`, `SwiGlu*` + `GatedMlp`, `Moe*` + `RunMoe`, `Mla*` +
  `RunMlaAttention`, `GatedDeltaNet*` + `RunGatedDeltaNet`, `NormConfig`/`RmsNorm`
  (`components/*.h`).
- Layer topology is `std::variant` dispatched by an if/else chain
  (`causal/decoder_layer.h:59-74`).
- Loading: `WeightNames`/`WeightLayout`, `LoadDecoderWeights`
  (`loading/weight_loader.h:41-45`); mmap + bf16/dequant in `Checkpoint`.
- State: `LayerStateSpec = variant<PagedKvStateSpec, RecurrentStateSpec>`,
  `ModelState{paged_kv, recurrent, layers[]}` (`state.h:18-55`).
- Config: generic `CheckpointConfig` + family-agnostic `DecoderConfig` + per-family
  translators.
- Quality scaffolding already present: parity harness with independent fp32
  reference per family, family config tests, parallel-linear shard tests, diagnostics
  trace.

This is a strong, correctness-first base. The issue is **extensibility of
composition and dispatch**, not correctness.

### 2.2 Gaps, with evidence

1. **God-object layer forward.** `DecoderLayer<Traits>::Forward` takes 15+
   parameters including every component's workspace (`decoder_layer.h:46-52`).
   Adding a component edits this signature.
2. **Dispatch is hard-coded.** Mixer selection is an if/else over `std::variant`
   (`decoder_layer.h:59-74`); a new mixer edits the variant, the layer, and possibly
   the workspace.
3. **Central workspace.** `DecoderWorkspace` aggregates buffers and every `max_*`
   dimension for all components (`decoder_stack.h:22-49`); components cannot size
   themselves.
4. **Traits are half-vestigial.** `ModelTraits` requires concrete `Norm/Attn/Mlp`
   types (`model_traits.h:17-27`), but runtime dispatch ignores them — hybrid models
   instantiate one type while layers vary (`families/qwen3.cc`). There is no
   per-layer component resolution.
5. **Weight loading is centralized and architecture-aware.** `LoadDecoderWeights`
   plus `WeightNames`/`WeightLayout` own all name mapping and sharding for every
   component. A new fused layout or a component-local parameter edits a shared
   loader.
6. **Registry is closed.** Adding a family touches `families.h`,
   `model_registry.cc`, and `src/models/CMakeLists.txt`; no plugin/registration path
   (`model_registry.cc:11-18`).
7. **Only one model shape.** Every family must be a `CausalLM<Traits>` (dense
   decoder stack). MTP heads, MTP-proposer modules, encoders, or structurally
   distinct models cannot compose — vLLM/SGLang treat these as ordinary child
   modules.
8. **State is a fixed two-way variant.** `ModelState`/`LayerStateSpec`
   (`state.h:18-55`) must be edited (and the runner, and layers) for latent MLA KV
   or convolution state, contradicting plan D19/D21.
9. **No capability descriptor.** The runner infers behavior from
   `StateRequirements()`; hybrid/MoE/recurrent facts aren't declared.
10. **Quantization is global and load-time-only** (`quant_config.h:20-40`); no
    per-module weight format adapter.

---

## 3. Proposed architecture

### 3.1 Principles

- **P1 — Compose, don't fork.** A model is a tree of modules; architectures are
  wiring + config parsing.
- **P2 — Small, typed execution interfaces.** Each module kind has one forward
  signature; the layer/stack never sees another component's workspace.
- **P3 — Cross-cutting state is an explicit `ForwardContext`.** No globals, no
  15-arg functions, no widening signatures.
- **P4 — Components are self-describing**: weights (`LoadWeights`), caches
  (`CacheRequirements`), workspaces (`Initialize`), and capabilities.
- **P5 — Names + declarative shard rules are the loading contract**, with a
  `WeightSource`/`WeightsMapper`/`ParamSpec` triple.
- **P6 — Caches are acquired as opaque handles**, so hybrid/recurrent/latent caches
  need no `ModelState` change.
- **P7 — Registries for models and components**, closed built-ins plus open
  out-of-tree registration.
- **P8 — Preserve the existing ops, error types, and the `Model`/`ModelRunner`
  serving boundary.**

### 3.2 Layering and dependency direction

```
engine / ModelRunner  ──depends on──►  models public API only
        │
   models::Model (facade)                       registry / builders
        │                                            │
   DecoderStack ─► DecoderBlock ─► Norm/Mixer/FeedForward ─► ops / cache / dist
        │
   loading (WeightSource, ParamSpec, shard rules) ─► Checkpoint
```

Allowed dependencies: `core`, `ops`, `cache` (abstract pools), `config`,
`dist::CommBackend` (passed in). Model code must not include `engine/`, `server/`,
or concrete `dist/` implementations. Enforcement: the model CMake target declares
only these deps (as `src/models/CMakeLists.txt:31-43` already does).

### 3.3 Target module layout

```
include/inferx/models/
  model.h                 # facade (kept name/signature)
  model_config.h          # ModelSpec, ModelDims, LayerSpec, Capabilities
  module.h                # Module base (params, children, prefix, caches, init)
  forward_context.h       # BatchGeometry, ParallelContext, ForwardContext, WorkspaceArena
  cache_spec.h            # CacheSpec, CacheHandle, CacheRegistry
  registry.h              # ModelRegistry, ModelBuilder, BuildOptions
  loading/
    weight_source.h       # WeightSource, WeightsMapper, ParamSpec, StackedParamSpec
    checkpoint.h          # unchanged (mmap, dequant)
    weight_names.h        # retained as a compatibility façade during migration
  components/
    norm.h
    mixer.h               # TokenMixer interface + MixerConfig + ComponentRegistry
    mixer_factory.h
    attention_mixer.h     # GqaMixer (was GqaAttention/RunAttention)
    mla_mixer.h
    linear_attention_mixer.h
    feed_forward.h        # FeedForward interface + FeedForwardConfig
    dense_mlp.h
    moe.h
  causal/
    decoder_block.h       # generic composed StandardDecoderBlock
    decoder_stack.h       # DecoderStack + LanguageModelHead
    causal_lm.h           # DenseDecoderModel : Model
    decoder_builder.h     # DenseDecoderBuilder : ModelBuilder
  families/               # per-architecture ModelSpec parsers + registration
  state.h                 # compatibility aliases (PagedKvState, RecurrentState, ModelState)
src/models/               # mirrors the above; one .cc per public header
```

### 3.4 Core type inventory

| Concept | Type | Replaces |
|---|---|---|
| One step's geometry | `BatchGeometry` | `AttentionBatch` + `ModelInput` |
| Cross-cutting env | `ForwardContext` | threaded workspace/variant args |
| Parallel view | `ParallelContext` | raw `ParallelConfig` + helpers |
| Scratch allocator | `WorkspaceArena` | `DecoderWorkspace` |
| Weight cursor | `WeightSource` | `Checkpoint` direct calls |
| Name rewrite | `WeightsMapper` | `WeightNames`/`WeightLayout` |
| Parameter binding | `ParamSpec`, `StackedParamSpec` | central shard branches |
| Cache requirement | `CacheSpec`, `CacheHandle` | `LayerStateSpec`/`ModelState` |
| Module lifecycle | `Module` | implicit traits wiring |
| Component interfaces | `Norm`, `TokenMixer`, `FeedForward` | `variant` + free functions |
| Executable layer | `DecoderBlock`, `StandardDecoderBlock` | `DecoderLayer<Traits>` |
| Model description | `ModelSpec`, `LayerSpec` | `DecoderConfig` |
| Capabilities | `ModelCapabilities` | none |
| Registration | `ModelRegistry`, `ModelBuilder` | `Family`, `ResolveFamily` |

---

## 4. Detailed API

Namespaces: public `inferx::models`; components `inferx::components`; decoder
`inferx::causal`. Error handling keeps `absl::Status`/`StatusOr` and the
`INFERX_*` macros.

### 4.1 Execution geometry and context

```cpp
namespace inferx {

/// One step's flat token batch and its cache addressing. Superset of the
/// current AttentionBatch, so the runner can keep constructing the same data.
struct BatchGeometry {
  Tensor token_ids;       // [num_tokens] int32
  Tensor positions;       // [num_tokens] int32
  Tensor batch_indices;   // [num_tokens] int32 (sequence index per token)
  Tensor qo_indptr;       // [num_seqs + 1] int32 (query offsets)
  Tensor kv_indptr;       // [num_seqs + 1] int32 (KV block counts)
  Tensor kv_indices;      // flattened block tables
  Tensor last_page_len;   // [num_seqs] int32
  Tensor logit_rows;      // [num_seqs] int32 (rows needing logits)
  absl::Span<const int32_t> host_qo_indptr;
  absl::Span<const int32_t> host_kv_indptr;
  std::optional<Tensor> cache_slots;  // per-cache slot indices (recurrent etc.)
  int num_tokens = 0;
  int num_seqs = 0;
};

/// Rank-local parallel view; owns no engine state.
struct ParallelContext {
  ParallelConfig config;
  int tp_rank = 0, tp_size = 1;
  int ep_rank = 0, ep_size = 1;
  dist::CommBackend* comm = nullptr;

  components::DimShard ShardRows(int64_t total) const;
  components::QkvParallelGeometry ShardQkv(const components::AttentionConfig&) const;
};

/// Named, step-stable scratch. Components allocate in Initialize and reuse.
class WorkspaceArena {
 public:
  StatusOr<Tensor> GetOrAlloc(std::string_view key, DataType dtype, Shape shape);
};

/// Everything a module needs that is not its own input/output.
struct ForwardContext {
  ops::OpContext& ops;
  const BatchGeometry& batch;
  const ParallelContext& parallel;
  CacheRegistry& caches;
  WorkspaceArena& workspace;
};

}  // namespace inferx
```

### 4.2 Module base, parameters, and weight binding

```cpp
namespace inferx::models {

/// Lifecycle + composition + weight/cache declaration. No execution method:
/// execution is defined by the specific mixin (Norm/TokenMixer/...).
class Module {
 public:
  virtual ~Module() = default;

  void SetPrefix(std::string prefix);
  const std::string& prefix() const;                  // "model.layers.3."

  /// Plan-time: acquire workspaces and cache handles.
  virtual Status Initialize(const PlanContext& plan) { return OkStatus(); }

  /// Recursive default loader over registered params/stacked params/children.
  virtual Status LoadWeights(const WeightSource& source, const WeightsMapper& mapper);

  /// Persistent cache this module (and descendants) require.
  virtual std::vector<CacheSpec> CacheRequirements() const { return {}; }

 protected:
  void RegisterParam(const ParamSpec& spec, Tensor* storage);
  void RegisterStackedParam(const StackedParamSpec& spec, Tensor* storage);
  void RegisterChild(std::string_view name, Module* child);
  void RegisterCacheSpec(CacheSpec spec);

  // Set by Initialize; modules read the pool through ctx.caches.Group(handle).
  std::optional<CacheHandle> AcquireCache(const PlanContext& plan, const CacheSpec& spec);
};

struct PlanContext {
  DeviceId device;
  ParallelContext parallel;
  int max_tokens = 0;
  int max_seqs = 0;
  WorkspaceArena* workspace = nullptr;
  CacheRegistry* caches = nullptr;
};

}  // namespace inferx::models
```

### 4.3 Components

```cpp
namespace inferx::components {

class Norm : public models::Module {
 public:
  virtual Status Forward(const Tensor& in, ops::OpContext&, Tensor& out) const = 0;
  virtual Status AddForward(const Tensor& in, Tensor& residual,
                            ops::OpContext&, Tensor& out) const = 0;
};

/// Token mixer (attention family). One instance per layer.
class TokenMixer : public models::Module {
 public:
  virtual Status Forward(const Tensor& normed, const models::ForwardContext& ctx,
                         Tensor& out) = 0;
  // CacheRequirements() inherited; e.g. PagedKv or Recurrent.
};

/// Feed-forward (dense MLP or MoE). One instance per layer.
class FeedForward : public models::Module {
 public:
  virtual Status Forward(const Tensor& normed, const models::ForwardContext& ctx,
                         Tensor& out) = 0;
};

// Concrete mixers: GqaMixer, MlaMixer, GdnMixer (one .h/.cc each).
// Concrete ffn:    DenseGatedMlp, MoeFeedForward.

/// Component construction is registry-driven so layers never switch on kind.
StatusOr<std::unique_ptr<Norm>>      MakeNorm(const NormConfig&, const models::PlanContext&);
StatusOr<std::unique_ptr<TokenMixer>> MakeTokenMixer(const MixerConfig&,
                                                     const models::PlanContext&);
StatusOr<std::unique_ptr<FeedForward>> MakeFeedForward(const FeedForwardConfig&,
                                                       const models::PlanContext&);

}  // namespace inferx::components
```

`MixerConfig` / `FeedForwardConfig` are `std::variant`s (attention/MLA/GDN;
SwiGLU/MoE) as today, but **only the factory consults them**. Adding a mixer = new
alternative + new factory case + class; `DecoderBlock`, `DecoderStack`, and
workspaces are untouched. For out-of-tree kinds, each variant carries an `External`
alternative wrapping `std::shared_ptr<ComponentConfigBase>` resolved through a
registry.

### 4.4 Decoder layer, stack, and head

```cpp
namespace inferx::causal {

struct ResidualState { Tensor hidden; Tensor scratch; };

/// One executed transformer layer. StandardDecoderBlock is the only concrete
/// implementation needed for dense/hybrid/MoE; exotic blocks subclass directly.
class DecoderBlock : public models::Module {
 public:
  virtual Status Forward(ResidualState& state, const models::ForwardContext& ctx) = 0;
};

/// Composed from Norm + TokenMixer + Norm + FeedForward; both residual styles
/// (pre-norm, Gemma output-norm) are config-driven inside this class.
class StandardDecoderBlock final : public DecoderBlock {
 public:
  StandardDecoderBlock(std::unique_ptr<components::Norm> input_norm,
                       std::unique_ptr<components::TokenMixer> mixer,
                       std::unique_ptr<components::Norm> post_norm,
                       std::unique_ptr<components::FeedForward> ffn,
                       LayerSpec spec);
  Status Forward(ResidualState& state, const models::ForwardContext& ctx) override;
  std::vector<models::CacheSpec> CacheRequirements() const override;
};

/// Embedding -> blocks -> final norm; returns last-row hidden states.
class DecoderStack : public models::Module {
 public:
  static StatusOr<std::unique_ptr<DecoderStack>> Create(ModelSpec spec,
                                                        const models::PlanContext&);
  Status Forward(const models::BatchGeometry& batch, const models::ForwardContext&,
                 Tensor& out_hidden);
  std::vector<models::CacheSpec> CacheRequirements() const override;
};

/// Gathers logit rows and projects to [num_seqs, vocab] (TP all-gather aware).
class LanguageModelHead : public models::Module {
 public:
  Status Forward(const Tensor& hidden, const models::BatchGeometry&,
                 const models::ForwardContext&, Tensor& logits);
};

}  // namespace inferx::causal
```

### 4.5 Model facade and capabilities

The serving boundary stays byte-compatible with `ModelRunner`:

```cpp
namespace inferx {

struct ModelCapabilities {
  bool is_hybrid = false;          // mixed attention/cache types
  bool has_recurrent_state = false;
  bool is_moe = false;
  bool is_attention_free = false;
  bool supports_pipeline_parallel = false;
  bool tie_word_embeddings = false;
  int64_t num_layers = 0;
  std::vector<CacheSpec> caches;
};

class Model {
 public:
  virtual ~Model() = default;

  static StatusOr<std::unique_ptr<Model>> Load(const std::string& directory, DeviceId device,
                                               int max_tokens, int max_seqs,
                                               const ParallelConfig& parallel = {});

  virtual const CheckpointConfig& config() const = 0;
  virtual const ModelCapabilities& capabilities() const = 0;

  /// Kept for source compatibility; forwards to the declared cache specs.
  virtual std::vector<LayerStateSpec> StateRequirements() const = 0;

  /// Unchanged runner contract; internally builds a ForwardContext.
  virtual StatusOr<Tensor> Forward(const ModelInput& input, ModelState& state,
                                   ops::OpContext& ctx, dist::CommBackend& comm) = 0;
};

}  // namespace inferx
```

`causal::DenseDecoderModel` is the standard `Model`: it owns a `DecoderStack` +
`LanguageModelHead`, and its `Forward` builds a `ForwardContext` from
`(ctx, comm, input.attention, parallel, cache_registry, workspace)`, runs the stack,
then the head. `ModelState` becomes a thin view over the `CacheRegistry` during the
migration (see §5).

### 4.6 Cache abstraction

```cpp
namespace inferx::models {

enum class CacheKind { kPagedKv, kLatentKv, kRecurrent, kConvolution };

struct CacheSpec {
  CacheKind kind = CacheKind::kPagedKv;
  int64_t layer = -1;                 // owning layer
  KvLayout layout;                    // paged/latent
  RecurrentSpec recurrent;            // recurrent/conv
  bool shareable_prefix = true;
};

struct CacheHandle { int64_t id = -1; };

/// Runner-owned registry; modules only see opaque handles.
class CacheRegistry {
 public:
  virtual CacheHandle Acquire(const CacheSpec&) = 0;
  virtual KvBlockPool* Paged(CacheHandle) const = 0;
  virtual RecurrentStatePool* Recurrent(CacheHandle) const = 0;
  virtual Status Bind(const CacheHandle&, BatchGeometry&) const = 0;  // fills indices
};

}  // namespace inferx::models
```

A `GqaMixer` acquires one `kPagedKv` handle; a `GdnMixer` acquires one `kRecurrent`
handle; an MLA mixer can acquire `kLatentKv`. Mixed models simply have a list of
specs, and `ModelCapabilities::is_hybrid` is derived. This is the plan-D19/D21
requirement satisfied structurally.

### 4.7 Weight loading

```cpp
namespace inferx::models {

/// Cursor over a checkpoint in device-agnostic bf16. Dequant is hidden here.
class WeightSource {
 public:
  virtual bool Contains(std::string_view name) const = 0;
  /// Full (unsharded) bf16 host tensor; borrows checkpoint storage.
  virtual StatusOr<Tensor> Host(std::string_view name, const Shape& expected_full) const = 0;
};

/// vLLM-style declarative remap: ordered substring/prefix/suffix replacements;
/// an empty replacement drops the tensor.
struct WeightsMapper {
  std::vector<std::pair<std::string, std::string>> substr, prefix, suffix;
  std::string Apply(std::string_view name) const;
};

enum class Parallelism { kReplicated, kColumn, kRow, kVocabParallel, kQkv, kMergedColumn };

/// One checkpoint tensor bound into a destination parameter.
struct ParamSpec {
  std::string name;                 // relative leaf, e.g. "q_proj.weight"
  Shape full_shape;                 // checkpoint shape, pre-sharding
  Parallelism parallelism = Parallelism::kReplicated;
  bool optional = false;
  // kQkv / kMergedColumn geometry:
  int64_t num_heads = 0, num_kv_heads = 0, head_dim = 0;
  std::vector<int64_t> merged_sizes;
};

/// A fused on-disk or runtime tensor (qkv_proj, gate_up_proj).
struct StackedParamSpec {
  std::string fused_name;
  Parallelism parallelism = Parallelism::kReplicated;
  struct Slot { std::string source; int64_t index; };  // {"q_proj.weight", 0}
  std::vector<Slot> slots;
  Shape fused_full_shape;
  int64_t num_heads = 0, num_kv_heads = 0, head_dim = 0;
};

}  // namespace inferx::models
```

`Module::LoadWeights` recursively: for each `ParamSpec`, compute
`mapper.Apply(prefix + name)`, require `source.Host(...)`, shard per `Parallelism`
(delegating to the existing `ShardQkv`/`ShardDim` helpers), then upload; for each
`StackedParamSpec`, fetch each slot and place it at its offset (KV-head replication
handled by `kQkv`); recurse into children with an extended prefix. A
`CheckpointWeightSource` adapts `models::Checkpoint` and preserves GPTQ/AWQ/MXFP4
dequant.

Retain `WeightNames`/`WeightLayout` as a **compatibility façade**: a family that
prefers explicit names can translate them into a `WeightsMapper` + `ParamSpec`s
without changing components.

### 4.8 Registry and construction

```cpp
namespace inferx::models {

struct BuildOptions {
  DeviceId device;
  int max_tokens = 0;
  int max_seqs = 0;
  ParallelConfig parallel;
};

/// Parses config into a ModelSpec and builds the module graph.
class ModelBuilder {
 public:
  virtual ~ModelBuilder() = default;
  virtual StatusOr<ModelSpec> Parse(const CheckpointConfig&, const nlohmann::json&) const = 0;
  virtual StatusOr<std::unique_ptr<Model>> Build(ModelSpec spec,
                                                 LoadedCheckpoint& checkpoint,
                                                 const BuildOptions&) const = 0;
};

struct ModelRegistration {
  std::string_view model_type;      // "qwen3"
  std::string_view architecture;    // "Qwen3ForCausalLM"
  StatusOr<std::unique_ptr<ModelBuilder>> (*make_builder)();
};

class ModelRegistry {
 public:
  static ModelRegistry& Instance();
  Status Register(ModelRegistration reg);          // built-ins + plugins
  StatusOr<const ModelRegistration*> Resolve(const CheckpointConfig&) const;
  StatusOr<std::unique_ptr<Model>> Build(LoadedCheckpoint&, const BuildOptions&) const;
};

// Helpers used by families that are dense decoders.
StatusOr<ModelSpec> ParseDenseDecoder(const CheckpointConfig&, const nlohmann::json&,
                                      const DenseDecoderDefaults&);
std::unique_ptr<ModelBuilder> MakeDenseDecoderBuilder();

}  // namespace inferx::models
```

Built-ins stay **explicitly registered** (static-link safe, as today), but
registration is one call per family and the `std::array` / `families.h` / CMake
enumeration is replaced by a single `RegisterBuiltins()` that `Register()`s each
`ModelRegistration`. Plugins may call `ModelRegistry::Instance().Register(...)` from
their own static initializer. `Model::Load` and `ModelRunner` are unchanged in
behavior.

### 4.9 Configuration

```cpp
namespace inferx::models {

struct ModelDims {         // replaces the generic half of DecoderConfig
  int64_t hidden_size, intermediate_size, num_hidden_layers;
  int64_t num_attention_heads, num_key_value_heads, head_dim;
  int64_t vocab_size, max_position_embeddings;
  float rms_norm_eps, rope_theta;
  bool tie_word_embeddings;
  // ...
};

struct LayerSpec {
  std::string type;                         // "full_attention" | "linear_attention" | ...
  components::MixerConfig mixer;            // variant
  components::FeedForwardConfig feed_forward;  // variant
  components::NormConfig input_norm, post_mixer_norm, mixer_out_norm, ffn_out_norm;
  components::ResidualStyle residual = components::ResidualStyle::kPreNorm;
};

struct ModelSpec {
  std::string model_type, architecture;
  ModelDims dims;
  components::NormConfig final_norm;
  float embedding_scale = 1.0f;
  int64_t embedding_row_offset = 0;
  std::vector<LayerSpec> layers;
  bool tie_word_embeddings = false;

  Status Validate() const;                       // == DecoderConfig::Validate
  Status ValidateExecutable() const;             // component availability
  ModelCapabilities Capabilities() const;
};

}  // namespace inferx::models
```

`ModelSpec` is the generalized `DecoderConfig`: same data, but layer components are
first-class and capabilities are derived rather than inferred.

### 4.10 Extension points

| To add… | Touch only… | Mechanism |
|---|---|---|
| New architecture | one `families/<name>.cc` + one `Register()` | `ModelBuilder` + `ModelRegistry` |
| New token mixer | `mixer.h` variant alt + `components/<name>.{h,cc}` + factory case | `TokenMixer` + `MakeTokenMixer` |
| New feed-forward | `feed_forward.h` alt + class + factory case | `FeedForward` |
| New cache kind | `cache_spec.h` + cache-layer group + runner sizing | `CacheSpec`/`CacheRegistry` |
| New residual/norm style | `Norm` impl or `StandardDecoderBlock` policy | `Norm` interface |
| New attention backend | `ops` backend registry | `AttentionBackend` (already) |
| New quant format | `loading/checkpoint.cc` + `WeightSource` impl | `WeightSource` |
| MTP/proposer head | attach a `Module` child; reuse `DecoderBlock` | module composition |
| New parallel axis | `ParallelContext` + `ParamSpec::Parallelism` | parallel helpers |

No engine, scheduler, or cache-manager changes are required for the first five.

---

## 5. Compatibility with the existing codebase

**Kept stable (no call-site changes):**

- `Model::Load`, `Model::config()`, `Model::StateRequirements()`,
  `Model::Forward(ModelInput, ModelState, OpContext&, CommBackend&)`.
- `ModelRunner` (`model_runner.h`), `SchedulerOutput`, `ModelRunnerOutput`,
  `KvBlockPool`, `RecurrentStatePool`, `OpContext`, `CommBackend`,
  `ParallelConfig`.
- `CheckpointConfig` and `CheckpointConfig::FromJson/FromFile` (used by
  server/CLI/tests).
- Ops free functions under `inferx::ops` and the `ops/cpu` + `ops/cuda` split.
- Error/Status macros.

**Type aliases / shims during migration:**

- `using AttentionBatch = BatchGeometry;`
- `ModelState` becomes a view: `paged_kv`/`recurrent` retained for tests; `layers`
  maps to acquired `CacheHandle`s.
- `WeightNames`/`WeightLayout` retained; internally converted to `WeightsMapper` +
  `ParamSpec`s.
- `causal::DecoderConfig` retained as `using DecoderConfig = ModelSpec;` (or
  `ModelSpec::ToDecoderConfig()` for the parity harness).

**Tests:**

- `tests/model_test.cc`, `model_runner_test.cc`, `tensor_parallel_test.cc` are
  unaffected if the facade and aliases hold.
- `tests/unit/models/parity_test.cc` should target `ModelSpec` (its `ReferenceModel`
  already consumes `causal::DecoderConfig`; an adapter suffices).
- `family_config_test.cc` and `parallel_linear_test.cc` keep working:
  `ShardQkv`/`ShardDim`/packed row layout are reused unchanged.
- `SafeTensorReader`/`Checkpoint`/quant tests unaffected.

**Static linking:** retain explicit registration (no reliance on static
initializers) with one `RegisterBuiltins()` translation unit; plugin registration is
additive.

---

## 6. Migration plan (no code now)

1. **Introduce types, no behavior change.** Add `BatchGeometry`, `ForwardContext`,
   `ParallelContext`, `WorkspaceArena`, `ModelSpec` (alias to `DecoderConfig`),
   `ModelCapabilities`.
2. **Wrap components as `Module`s.** Port `RmsNorm`, `GatedMlp`, `RunMoe`,
   `RunAttention`, `RunMlaAttention`, `RunGatedDeltaNet` behind
   `Norm`/`TokenMixer`/`FeedForward`, calling the existing ops unchanged; add
   factories.
3. **Replace the god layer.** Implement `StandardDecoderBlock` and `DecoderStack` on
   the new interfaces, keeping bit-identical arithmetic order. Prove equivalence with
   the parity harness.
4. **Move weight loading to `WeightSource` + `ParamSpec`.** Keep
   `WeightNames`/`WeightLayout` as a shim; convert families one at a time.
5. **Introduce `ModelRegistry`/`ModelBuilder`.** Register built-ins; make
   `Model::Load` delegate.
6. **Switch cache state to handles.** Introduce `CacheRegistry`; make `ModelState` a
   view; enable a second cache kind (latent MLA or recurrent) end-to-end.
7. **Add an out-of-tree example** (e.g., a Qwen3-Next layer via `Register`) to
   validate the extension path.
8. **Delete shims** once no call sites remain.

Each step is independently verifiable by the existing parity and family-config
tests.

---

## 7. Risks and open questions

- **Virtual dispatch per module per layer.** Cost is O(layers) calls per step
  (≈28), negligible; the existing no-virtual fast path inside ops is preserved.
  Worth a micro-benchmark at step 3.
- **`ModelState` view semantics** under handle-based caches; the runner currently
  owns pools and passes them in — need a clean ownership story for `CacheRegistry`.
- **Workspace keyed allocation** adds a map lookup at init only; confirm no per-step
  lookups.
- **Variant vs. open component configs.** Keeping `std::variant` preserves closed-set
  ergonomics; the `External` alternative is the escape hatch. Decide whether that is
  needed before step 2.
- **`ModelSpec` ↔ `DecoderConfig` aliasing** may be too loose for the parity
  harness; prefer an explicit converter.
- **How far to push `Module` reflection.** This proposal uses explicit registration
  calls in constructors (not Python-style reflection), to keep C++ predictable.
