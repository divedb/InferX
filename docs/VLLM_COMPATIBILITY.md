# vLLM compatibility matrix

InferX's CLI and HTTP API track **vLLM 0.26.0** as the compatibility baseline.
The contract is: a launch script or client written for vLLM either works
identically against `inferx-serve`, or fails **fast and loud at startup (or
with a clear HTTP 400)** naming the vLLM feature InferX does not provide.
Nothing silently degrades. `inferx-serve --help` is the authority for the full
grouped flag surface; this file is the map of what is real, what is stubbed,
and where behavior deliberately diverges.

The old InferX flag spellings still parse as hidden deprecated aliases, so
existing InferX scripts keep working; new material should use the vLLM names.

## Implemented serve flags

Flags that select real InferX behavior. Defaults are InferX's; where vLLM
derives a default from the model config (e.g. `--max-model-len`), InferX uses
the fixed default listed here instead.

| vLLM flag | old InferX alias | default | notes |
|---|---|---|---|
| `model_tag` (positional) | — | — | checkpoint directory, like `vllm serve <dir>`; equivalent to `--model` |
| `--model` | (same) | — | checkpoint directory (local only) |
| `--served-model-name` | (same) | checkpoint directory basename | name reported in responses; a single alias only — multiple aliases (which vLLM allows) are a startup error |
| `--dtype` | — | `auto` | only `auto` and `bfloat16` are accepted; `half`/`float16`/`float`/`float32` parse but error at startup (InferX runs bf16 only) |
| `--max-num-seqs` | `--max-running` | 8 | concurrent sequences |
| `--max-model-len` | `--max-seq-len` | 2048 | prompt + generation cap |
| `--max-num-batched-tokens` | — | unset | token budget per scheduler step; unset derives `max(2048, max-model-len)` |
| `--enable-prefix-caching` / `--no-enable-prefix-caching` | — | enabled | reuse prompt-prefix KV across requests |
| `--enable-chunked-prefill` | — | enabled | always on; `--no-enable-chunked-prefill` is a startup error |
| `--num-gpu-blocks-override` | `--kv-blocks` | unset (auto) | pins the KV pool to an exact block count. **Default changed**: unset now auto-sizes (see below); the old `--kv-blocks` default was a fixed count |
| `--block-size` | (same) | 16 | tokens per KV block |
| `--gpu-memory-utilization` | — | 0.92 | fraction of GPU memory the engine may use; drives KV auto-sizing |
| `--kv-cache-memory-bytes` | — | unset | explicit KV cache budget in bytes |
| `--tensor-parallel-size` | (same) | 1 | 1 or 2 |
| `--device-ids` | `--devices` | 0 | comma-separated CUDA devices |
| `--quantization` / `-q` | `--fp8`, `--w4a16` | none | `fp8` = weight fp8 (was `--fp8`); `w4a16` is an **InferX extension value** (was `--w4a16`) |
| `--kv-cache-dtype` | `--fp8-kv` | `auto` | choices: `auto`, `bfloat16`, `fp8`, `fp8_e4m3` |
| `--enforce-eager` / `--no-enforce-eager` | `--no-cuda-graphs` | off | skip decode CUDA-graph capture |
| `--host` / `--port` | (same) | 127.0.0.1 / 8000 | bind address |
| `--api-key` | — | none | plaintext bearer token, repeatable; **hashed at startup**, never stored raw. `--api-key-sha256` remains as the InferX extension for pre-hashed keys |
| `--config <file>` | — | none | **TOML/INI only — vLLM YAML config files are rejected** with an error; translate the keys |
| `--log-level`, `--v`, `--log-json`, `--log-file` | (same) | info | logging |

InferX-extension flags (no vLLM counterpart), grouped as "InferX extensions"
in `--help`: `--comm-backend {single,nccl}`,
`--collective-timing-sample-rate`, `--api-key-sha256`,
`--scheduler-endpoint`, and the HTTP knobs `--max-active-requests` (1024),
`--max-request-bytes` (8 MiB), `--read-timeout` (30 s), `--write-timeout`
(600 s), `--request-timeout` (600 s), `--io-threads` (0 = auto),
`--application-threads` (0 = auto).

## KV cache sizing

vLLM sizes its KV cache from a memory fraction; InferX now does the same. With
nothing pinned, after weights load the engine measures free VRAM and sizes the
KV pool so total engine usage stays within `--gpu-memory-utilization`
(default 0.92). Precedence when several knobs are set:

1. `--num-gpu-blocks-override <N>` — exact block count, beats everything;
2. `--kv-cache-memory-bytes <B>` — explicit byte budget;
3. `--gpu-memory-utilization <F>` — the auto-size default.

Scripts that previously relied on the old `--kv-blocks <N>` fixed default
should pass `--num-gpu-blocks-override <N>` to keep identical sizing.

## Compat stubs: parse everything, act on nothing silently

The remaining vLLM 0.26.0 serve surface — 252 flags, grouped in `--help` under
"vLLM compatibility (defaults only)" — parses but selects no behavior. Any
value **other than the vLLM default** is a startup error that names the
feature, so a vLLM launch script only starts if InferX will actually do what
the script asked for. Real examples:

```text
inferx-serve: --pipeline-parallel-size=2 requests pipeline parallelism, which is not supported by InferX (only the vLLM default '1' is accepted)
inferx-serve: --enable-lora=true requests LoRA adapters, which is not supported by InferX (only the vLLM default 'false' is accepted)
inferx-serve: --speculative-config=x requests speculative decoding, which is not supported by InferX
```

Flags newer than the 0.26.0 table fail to parse outright — the correct loud
failure for an unknown flag.

### Accept-any list

A small set of stub flags is accepted with **any** value, because they ask for
less than InferX already provides or only shape logs vLLM would write. Each
logs a one-line note (`accepted for vLLM compatibility; it has no effect on
InferX`) and does nothing:

`--aggregate-engine-logging`, `--disable-log-stats`, `--enable-log-requests`,
`--fail-on-environ-validation`, `--uvicorn-log-level`,
`--disable-uvicorn-access-log`, `--disable-access-log-for-endpoints`,
`--log-config-file`, `--max-log-len`, `--log-error-stack`,
`--disable-fastapi-docs`, `--enable-offline-docs`, `--enable-log-outputs`,
`--enable-log-deltas`, `--trust-remote-code`, `--use-tqdm-on-load`,
`--disable-custom-all-reduce`, `--enable-logging-iteration-details`.

## Gateway (`inferx-gateway`) differences

The gateway shares the frontend-only slice of the policy:

- gained `--api-key` (plaintext, hashed at startup) alongside the existing
  `--api-key-sha256` extension;
- carries the **frontend** compat stubs only (CORS, TLS, log shaping, …) with
  the same defaults-only rule — engine flags are not its business;
- `--chat-template` takes a **built-in template name**: `qwen2` or
  `deepseek-v2`. vLLM-style Jinja template file paths are a startup error
  (`--chat-template files (Jinja) are not supported by InferX; use a built-in
  template name: qwen2, deepseek-v2`).

## Per-request API parameters

> **Deliberate divergence — sampling default.** InferX's default `temperature`
> is **0.0 (greedy)**, not vLLM/OpenAI's 1.0. A request that sends no sampling
> parameters gets deterministic output. Pass `temperature` explicitly for
> vLLM-identical behavior. This is intentional and will not change silently.

Unknown JSON fields are **ignored** (OpenAI parity); known-but-unsupported
fields get a clear 400.

### Implemented

| parameter | notes |
|---|---|
| `temperature` | default **0.0** (see divergence above) |
| `top_p`, `top_k`, `min_p` | |
| `presence_penalty`, `frequency_penalty`, `repetition_penalty` | windowed: computed over the last 256 unique generated tokens |
| `n` | non-streaming only; capped |
| `seed` | per-request |
| `stop`, `stop_token_ids` | |
| `ignore_eos`, `min_tokens` | `min_tokens` masks at most 16 stop token ids |
| `skip_special_tokens`, `include_stop_str_in_output` | |
| `logprobs`, `top_logprobs` | chat bool + int spelling and completions int spelling; `top_logprobs` ≤ 20 |
| `echo` | completions only |
| `max_tokens`, `max_completion_tokens` | alias accepted |
| `stream`, `stream_options.include_usage` | |

### Rejected with a clear 400

`logit_bias`; structured outputs (`response_format` other than text, all
`guided_*`); `tools` / `tool_choice` other than `none`; beam search;
`best_of` ≠ `n`; `bad_words`; `allowed_token_ids`; `prompt_logprobs`;
`truncate_prompt_tokens`; `prompt_embeds`; `suffix`;
`spaces_between_special_tokens: false`; `echo` on chat; streaming with
`n > 1`.

## Endpoints

| endpoint | status |
|---|---|
| `/v1/chat/completions` | present |
| `/v1/completions` | present |
| `/v1/models` | present |
| `/v1/embeddings` | present |
| `/v1/tokenize` | present (InferX-style) |
| `/health*` | present |
| `/metrics` | present (Prometheus) |
| `/stats` | present (legacy InferX) |
| `/v1/models/{id}` | in progress |
| `/version`, `/ping` | in progress |
| `/tokenize`, `/detokenize` (vLLM-style) | in progress |
| `/pooling`, `/score`, `/rerank` | in progress — will return 501 |

## Known limitations

- Tensor parallelism is TP ≤ 2 (`--tensor-parallel-size 1|2`).
- bf16 weights only (`--dtype auto|bfloat16`); `--quantization fp8|w4a16` and
  `--kv-cache-dtype fp8` quantize from a bf16 checkpoint.
- `n > 1` is non-streaming only, and capped.
- Repetition/presence/frequency penalties are windowed to the last 256 unique
  generated tokens rather than the full history.
- `min_tokens` masks at most 16 stop token ids.
- `top_logprobs` ≤ 20.
- `--config` reads TOML/INI, not vLLM's YAML.
