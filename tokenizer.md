# Tokenizer architecture

Status: implemented initial architecture. Hugging Face `tokenizer.json` and
native SentencePiece artifacts are selected through the pinned
`mlc-ai/tokenizers-cpp` backend. Independent backend clones are available for
request-scoped ownership. The optional sidecar and chat-template extraction
remain future work.

## 1. Why the current design does not scale

The former `Tokenizer` was a correct, deliberately narrow implementation of two
byte-level BPE pipelines: Qwen2 and o200k. It parses a Hugging Face
`tokenizer.json`, but it does not interpret that file's tokenizer graph. Instead,
it recognizes two exact split expressions and implements their normalization,
pre-tokenization, byte mapping, BPE, and decoding in C++.

That approach has served the two supported checkpoints, but a new tokenizer may
require another regex, normalizer sequence, added-token rule, post-processor,
decoder, or tokenization model. Supporting it then means adding model-specific
logic to a central class and maintaining our own Unicode and regex compatibility.
Conformance is demonstrated for a sample corpus rather than inherited from the
implementation that created the checkpoint. The cost and risk grow with every
model.

InferX should own a small serving interface and checkpoint policy, not its own
general-purpose tokenization algorithms.

## 2. Goals and non-goals

The design must:

- load tokenizers from checkpoint artifacts rather than model-name conditionals;
- cover byte BPE, WordPiece, Unigram/SentencePiece, WordLevel, and future
  pipelines supported by an installed backend;
- preserve the checkpoint's normalizer, pre-tokenizer, added-token,
  post-processor, and decoder semantics exactly;
- distinguish trusted control text from untrusted user text;
- support concurrent encode/decode and exact incremental decoding;
- validate tokenizer IDs and special-token metadata against the model;
- fail at load time, with an actionable error, when semantics are unsupported;
- make backend upgrades observable, pinned, and conformance-tested.

It is not a goal to train tokenizers, silently approximate an unsupported
pipeline, infer behavior from a model family name, or execute arbitrary
checkpoint code in the InferX server process.

## 3. Proposed architecture

```text
checkpoint directory
        |
        v
TokenizerLoader ---- reads/validates artifacts and special-token metadata
        |
        v
BackendRegistry ---- probes factories in priority order; returns diagnostics
        |
        +---- TokenizersCppBackend/HuggingFace (`tokenizer.json`, default)
        +---- TokenizersCppBackend/SentencePiece (`tokenizer.model`, compatibility)
        +---- OptionalSidecarBackend (explicit opt-in escape hatch)
        |
        v
TokenizerService ---- stable InferX API, security policy, validation, metrics
        |
        +---- immutable SharedTokenizer (safe across request threads)
        +---- per-request DecodeStream (mutable and never shared)
```

There are four separate responsibilities:

1. `TokenizerLoader` discovers artifacts and produces validated configuration.
2. `BackendRegistry` selects an implementation by artifact and capability, not
   by model name.
3. `TokenizerService` provides the only tokenizer API used by the server.
4. Each backend owns the complete algorithm and its streaming decoder state.

This separation keeps engine, HTTP, and scheduler code independent of the
third-party library and allows a backend to be replaced without changing call
sites.

## 4. Stable InferX interface

The public interface should describe operations and policy rather than BPE
internals. The following is illustrative, not a committed C++ header:

```cpp
using TokenId = int32_t;

enum class SpecialTokenMode {
  kAsControl,  // recognize configured added/special tokens in trusted text
  kAsText,     // never let input text inject a control token
};

struct EncodeOptions {
  SpecialTokenMode special_tokens = SpecialTokenMode::kAsText;
  bool add_post_processor_tokens = false;
};

struct TokenizerInfo {
  std::string backend;
  std::string backend_version;
  std::string artifact_sha256;
  int64_t vocabulary_size;
  std::optional<TokenId> bos_id;
  std::optional<TokenId> eos_id;
  std::optional<TokenId> pad_id;
  std::vector<TokenId> stop_ids;
  bool supports_incremental_decode;
};

class DecodeStream {
 public:
  virtual StatusOr<std::string> Push(TokenId id) = 0;
  virtual StatusOr<std::string> Finish() = 0;
};

class Tokenizer {
 public:
  virtual StatusOr<std::vector<TokenId>> Encode(
      std::string_view text, const EncodeOptions&) const = 0;
  virtual StatusOr<std::string> Decode(
      std::span<const TokenId> ids, bool skip_special) const = 0;
  virtual StatusOr<std::unique_ptr<DecodeStream>> StartDecode(
      bool skip_special) const = 0;
  virtual std::optional<TokenId> TokenToId(std::string_view token) const = 0;
  virtual StatusOr<std::string> IdToToken(TokenId id) const = 0;
  virtual bool IsSpecial(TokenId id) const = 0;
  virtual const TokenizerInfo& Info() const = 0;
};
```

All fallible InferX-specific operations return `StatusOr`. Invalid UTF-8 policy,
invalid IDs, unsupported options, and backend failures must not become an
exception crossing the ABI. Tokenizer configuration is immutable after loading,
but the current tokenizers-cpp handle is request-local because its FFI mutates
handle-owned decode and lookup buffers. A `DecodeStream` is request-local too.

The existing convenient distinction between `Encode` and `EncodeOrdinary`
should remain at API call sites, but map explicitly to `SpecialTokenMode`.
Untrusted API input defaults to `kAsText`; only prompt text produced by a trusted
template renderer may use `kAsControl`.

Batch encode can be added without changing semantics. Offsets, token strings,
type IDs, and attention masks should be optional capabilities returned through
an `Encoding` type if an embeddings or classification endpoint needs them; the
generation path should not pay for them by default.

## 5. Backend selection

Factories register the artifact types and capabilities they understand. Loading
uses deterministic priority:

1. A complete `tokenizer.json` uses the Hugging Face backend.
2. If no usable JSON exists, `tokenizer.model` or `spiece.model` uses the native
   SentencePiece backend.
3. An optional sidecar backend may be selected only by operator configuration.

The loader must not fall through after a malformed higher-priority artifact. A
broken `tokenizer.json` beside a valid SentencePiece file is probably a damaged
checkpoint and should produce an error unless the operator explicitly selects
the other backend. An override such as `--tokenizer-backend=sentencepiece` is
useful for diagnosis and controlled rollout; automatic model-name tables are
not.

Every factory returns either a loaded tokenizer or a structured rejection that
names the artifact, unsupported component/type, backend version, and remedy.
Unknown JSON graph nodes are rejected rather than ignored.

## 6. Recommended third-party integration

### 6.1 Default: `mlc-ai/tokenizers-cpp`

The implemented backend is `mlc-ai/tokenizers-cpp`, pinned as a git submodule.
It wraps the upstream Rust `tokenizers` engine and the official SentencePiece
C++ library behind a common C++ interface. InferX's `Tokenizer` is a thin owning
adapter: `Encode`, `EncodeBatch`, `Decode`, `GetVocabSize`, `IdToToken`, and
`TokenToId` forward directly to `tokenizers::Tokenizer`; InferX-specific checked
operations, metadata, special-token policy, and cloning are layered alongside
that surface. Hugging Face still models tokenization as a configurable
pipeline of normalizer, pre-tokenizer, model, post-processor, and decoder
components and supports BPE, WordPiece, Unigram, and WordLevel. Loading the
checkpoint graph gives exactness by construction for supported components.

The submodule owns its Rust C ABI and static-library build. Its checkpoint-aware
Hugging Face factory accepts both `tokenizer.json` and `tokenizer_config.json`
and exposes the resolved BOS, EOS, and PAD IDs. InferX validates those IDs and
performs special-token policy above the common interface. `Tokenizer::Clone()` recreates
an independent backend handle from the already-loaded artifact and copies the
resolved metadata, allowing each concurrent request to own its mutable decode
and lookup buffers without locking. Tokenizer instances are request-confined;
HTTP handlers acquire a clone and incremental decoders borrow the owning
request's tokenizer. A future upstream API extension should add structured
load/runtime errors, a cheaper native clone, direct ordinary-text mode, and
direct decode-stream access.

The current adapter handles ordinary user text by identifying added-token
spellings from `tokenizer.json` and encoding those spellings in pieces, preventing
multi-character control-token injection. Single-codepoint added tokens are
reported as unsupported by the fallible API because the common upstream API
cannot disable added-token recognition. Post-processor insertion is likewise an
explicit unsupported option until exposed upstream.

### 6.2 Native SentencePiece compatibility backend

Some checkpoints ship only a serialized SentencePiece model. The official C++
SentencePiece library is appropriate for these artifacts and supports its BPE
and Unigram models without reconstructing them in InferX. It remains a separate
backend because a bare SentencePiece model does not carry every Transformers
fast-tokenizer behavior, such as arbitrary post-processing and added-token
rules. Companion metadata must be applied only where its meaning is explicit
and tested.

If `tokenizer.json` embeds a Unigram/SentencePiece-like pipeline, the Hugging
Face backend remains preferred because it interprets the whole graph.

### 6.3 Optional sidecar for exceptional tokenizers

There will be checkpoints whose tokenizer exists only as a Transformers "slow"
Python class or custom repository code. A versioned, length-prefixed sidecar
protocol can provide an explicit compatibility escape hatch without importing
Python or executing checkpoint code in the inference process.

This backend is disabled by default. If enabled, the sidecar must be pinned,
sandboxed, resource-limited, started and health-checked by the operator, and
forbid remote code unless separately authorized. Its protocol implements the
same interface and reports its exact package/model revisions. It is not an
automatic fallback: silently sending prompts to an arbitrary process is both a
security and operability problem.

## 7. Checkpoint metadata and validation

Tokenizer behavior is assembled from artifacts, never guessed from architecture
names. The loader should recognize, as applicable:

- `tokenizer.json`;
- `tokenizer.model` or `spiece.model`;
- `tokenizer_config.json`;
- `special_tokens_map.json`;
- `added_tokens.json`;
- model `config.json` for vocabulary dimensions and declared token IDs.

One metadata resolver establishes BOS, EOS, PAD, UNK, added/special tokens, and
generation stop IDs with documented precedence. Conflicting IDs or token
spellings are load errors. The resolved metadata is immutable and exposed in
`TokenizerInfo`.

At load time validate that:

- every declared token ID is non-negative, representable by `TokenId`, and
  resolvable by the backend;
- tokenizer IDs fit the model's input embedding rows (the tokenizer vocabulary
  need not equal a padded model `vocab_size`);
- required EOS/stop tokens exist and special-token classification agrees;
- encode/decode smoke tests succeed, including empty, ASCII, multilingual,
  combining-mark, emoji, whitespace, and embedded-NUL inputs;
- streaming is enabled only if its conformance contract is satisfied.

Chat templating is a separate layer above tokenization. A `ChatTemplateRenderer`
turns typed messages into trusted text or, eventually, token IDs. It may use a
safe supported subset of the checkpoint template format or a registered
operator-supplied renderer. It must not be hardcoded inside a tokenizer backend,
and arbitrary Jinja or checkpoint code must not execute in the server process.

## 8. Incremental decoding contract

Incremental decoding is not generically equivalent to decoding each ID alone.
Tokens may split UTF-8 characters, and decoder chains can join or clean text
based on neighboring tokens. Therefore `TokenizerService` must not implement a
universal per-token `IdToToken` decoder.

Each backend supplies or is adapted to a stateful stream and may buffer bytes or
token IDs until output is stable. The required invariant is:

```text
concatenate(Push(id) for every id) + Finish()
    == Decode(all_ids, skip_special)
```

Every non-final chunk must be valid UTF-8, no previously emitted bytes may be
revised, special tokens must obey the selected policy, and invalid IDs must
return errors. A backend that cannot meet the invariant declares incremental
decode unsupported; the server then rejects streaming for that tokenizer or
buffers the entire completion according to explicit endpoint policy. It must
never silently use a lossy approximation.

The initial `tokenizers-cpp` adapter cumulatively decodes the token prefix,
emits only the UTF-8-aligned common prefix with the preceding decode, and flushes
the retained suffix at completion. This provides a generic implementation and
passes exact conformance for Qwen2 and o200k. Exposing Hugging Face's native
`DecodeStream` through `tokenizers-cpp` remains the preferred optimization and
correctness generalization before declaring arbitrary new decoder graphs
streaming-capable.

## 9. Concurrency, caching, and performance

Loaded tokenizer data is immutable and shared across request threads. Mutable
decoder state is per request. If a third-party handle is not documented as safe
for concurrent calls, the backend uses a small pool of cloned handles rather
than a global mutex.

Batching belongs in `TokenizerService`, independently of inference batching. It
may use a bounded CPU worker pool and bounded queues so prompt floods apply
backpressure. A cache is optional and backend-neutral; if used, its key includes
the artifact digest, backend version, full encode options, and input bytes. It
must be bounded by bytes, expose hit/eviction metrics, and never mix trusted
control mode with ordinary-text mode.

Measure load time, encode/decode latency, input bytes, output token counts,
queueing, errors by backend and reason, and sidecar restarts. Do not put raw
prompt text or token content in logs or metric labels.

## 10. Compatibility and test strategy

Correctness is exact ID equality, not approximate round-tripping. For every
supported checkpoint fixture, tests should compare InferX against the canonical
reference for:

- encode in ordinary and control-token modes;
- batch encode and single encode;
- decode with and without special tokens;
- token/ID lookup and resolved metadata;
- incremental output concatenated against batch decode;
- malformed UTF-8 policy and invalid IDs;
- Unicode normalization, scripts, combining marks, emoji, whitespace,
  punctuation, long input, and added-token overlap.

Keep small golden corpora in the repository and run a broader differential suite
in CI against pinned upstream tools. Generate randomized byte/Unicode cases and
record the seed on failure. Each dependency upgrade runs the full model matrix
and publishes behavior and performance deltas; tokenizer artifacts and backend
versions are part of reproducibility metadata.

Loader tests should cover missing, ambiguous, truncated, oversized, and unknown
artifacts. Fuzz the JSON/FFI boundary and cap artifact size, vocabulary size,
input size, output expansion, recursion depth, and sidecar response size.

Initial compatibility targets should deliberately span pipeline families:
Qwen2 and o200k byte BPE, a Llama/T5-style SentencePiece tokenizer, BERT
WordPiece, and a Unigram tokenizer. Adding a model whose existing backend
understands its artifacts should require a fixture and metadata validation, not
a new algorithm or model-family branch.

## 11. Migration plan

1. **Done:** introduce the stable interface, artifact loader, metadata resolver,
   and `tokenizers-cpp` adapter.
2. **Done:** make Hugging Face the default for `tokenizer.json` and retain exact
   Qwen2 and o200k differential corpora.
3. **Done:** select native SentencePiece for checkpoints that lack
   `tokenizer.json`.
4. Add representative checked-in WordPiece, Unigram, and SentencePiece fixtures
   independent of locally cached production checkpoints.
5. Expose native ordinary-text and `DecodeStream` operations upstream, replacing
   the adapter fallbacks.
6. Move chat-template selection out of model-specific HTTP code and into the
   separate renderer boundary.
7. Remove the remaining handwritten Unicode compatibility tables when no tests
   or streaming compatibility paths require them.
8. Add the sidecar only when a concrete required checkpoint cannot be expressed
   by the in-process backends.

During migration, operators should be able to log the selected backend and
artifact digest and force a backend for canary comparison. No request should
choose a backend dynamically; selection happens once when the model is loaded.

## 12. Decision summary

InferX now uses a backend-oriented tokenizer service with artifact-driven
selection. The pinned `mlc-ai/tokenizers-cpp` submodule supplies the upstream
Hugging Face Rust engine for `tokenizer.json` and official SentencePiece C++ for
SentencePiece-only checkpoints. All third-party types remain behind the InferX
interface, unsupported semantics are explicit, and a sandboxed sidecar remains
reserved for exceptional future cases.

This changes the unit of model support from “write another tokenizer in InferX”
to “prove that a backend interprets this checkpoint exactly,” which is the
scalable path for existing and future models.

## References

- [Hugging Face tokenizer pipeline and components](https://huggingface.co/docs/tokenizers/main/components)
- [Hugging Face `tokenizers` source and supported bindings](https://github.com/huggingface/tokenizers)
- [`mlc-ai/tokenizers-cpp`](https://github.com/mlc-ai/tokenizers-cpp)
- [SentencePiece](https://github.com/google/sentencepiece)
