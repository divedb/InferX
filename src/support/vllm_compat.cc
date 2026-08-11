#include "inferx/support/vllm_compat.h"

#include <openssl/evp.h>

#include <array>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <sstream>

#include "CLI/CLI.hpp"
#include "absl/log/log.h"
#include "absl/strings/str_cat.h"

namespace inferx {

namespace {

constexpr std::string_view kGroup = "vLLM compatibility (defaults only)";

// The vLLM 0.26.0 flag surface that InferX parses but cannot act on. Grouped
// as vLLM's --help groups them. `default_text` is vLLM's default rendered as
// a literal; empty means vLLM's None, for which any explicit value names a
// feature InferX lacks. Rows with accept_any ask for less than InferX already
// provides (or only shape logs vLLM would write) and are accepted with a note.
//
// Flags absent from this table and from the real option set fail to parse,
// which is the correct loud failure for vLLM flags newer than this table.
constexpr CompatStub kServeStubs[] = {
    // vllm serve top-level
    {"--aggregate-engine-logging", CompatKind::kBool, "false", "log shaping",
     true},
    {"--api-server-count", CompatKind::kInt, "1",
     "multiple API-server processes"},
    {"--disable-log-stats", CompatKind::kBool, "false", "log shaping", true},
    {"--enable-log-requests", CompatKind::kBool, "false", "log shaping", true},
    {"--fail-on-environ-validation", CompatKind::kBool, "false",
     "vLLM environment validation", true},
    {"--gdn-prefill-backend", CompatKind::kString, "",
     "GDN prefill backend selection"},
    {"--grpc", CompatKind::kBool, "false", "the gRPC frontend"},
    {"--headless", CompatKind::kBool, "false", "headless data-parallel nodes"},
    {"--shutdown-timeout", CompatKind::kInt, "0", "shutdown-timeout override"},

    // Frontend
    {"--uds", CompatKind::kString, "", "unix-domain-socket binding"},
    {"--root-path", CompatKind::kString, "", "reverse-proxy root paths"},
    {"--middleware", CompatKind::kString, "", "custom ASGI middleware"},
    {"--allow-credentials", CompatKind::kBool, "false", "CORS configuration"},
    {"--allowed-origins", CompatKind::kString, "", "CORS configuration"},
    {"--allowed-methods", CompatKind::kString, "", "CORS configuration"},
    {"--allowed-headers", CompatKind::kString, "", "CORS configuration"},
    {"--ssl-keyfile", CompatKind::kString, "", "TLS termination"},
    {"--ssl-certfile", CompatKind::kString, "", "TLS termination"},
    {"--ssl-ca-certs", CompatKind::kString, "", "TLS termination"},
    {"--ssl-cert-reqs", CompatKind::kInt, "0", "TLS client certificates"},
    {"--ssl-ciphers", CompatKind::kString, "", "TLS termination"},
    {"--enable-ssl-refresh", CompatKind::kBool, "false", "TLS cert refresh"},
    {"--uvicorn-log-level", CompatKind::kString, "info", "log shaping", true},
    {"--disable-uvicorn-access-log", CompatKind::kBool, "false", "log shaping",
     true},
    {"--disable-access-log-for-endpoints", CompatKind::kString, "",
     "log shaping", true},
    {"--log-config-file", CompatKind::kString, "", "uvicorn log config",
     true},
    {"--max-log-len", CompatKind::kInt, "", "log shaping", true},
    {"--log-error-stack", CompatKind::kBool, "false", "log shaping", true},
    {"--disable-fastapi-docs", CompatKind::kBool, "false", "FastAPI docs",
     true},
    {"--enable-offline-docs", CompatKind::kBool, "false", "FastAPI docs",
     true},
    {"--h11-max-incomplete-event-size", CompatKind::kInt, "4194304",
     "h11 tuning"},
    {"--h11-max-header-count", CompatKind::kInt, "256", "h11 tuning"},
    {"--enable-request-id-headers", CompatKind::kBool, "false",
     "request-id response headers"},
    {"--chat-template", CompatKind::kString, "", "custom chat templates"},
    {"--chat-template-content-format", CompatKind::kString, "auto",
     "chat-template content formats"},
    {"--trust-request-chat-template", CompatKind::kBool, "false",
     "request-supplied chat templates"},
    {"--default-chat-template-kwargs", CompatKind::kString, "",
     "chat-template kwargs"},
    {"--response-role", CompatKind::kString, "assistant",
     "a non-default response role"},
    {"--return-tokens-as-token-ids", CompatKind::kBool, "false",
     "tokens-as-token-ids output"},
    {"--lora-modules", CompatKind::kString, "", "LoRA adapters"},
    {"--enable-auto-tool-choice", CompatKind::kBool, "false", "tool calling"},
    {"--exclude-tools-when-tool-choice-none", CompatKind::kBool, "false",
     "tool calling"},
    {"--tool-call-parser", CompatKind::kString, "", "tool calling"},
    {"--tool-parser-plugin", CompatKind::kString, "", "tool calling"},
    {"--tool-server", CompatKind::kString, "", "tool calling"},
    {"--enable-prompt-tokens-details", CompatKind::kBool, "false",
     "prompt-token details in usage"},
    {"--enable-per-request-metrics", CompatKind::kBool, "false",
     "per-request metrics"},
    {"--enable-server-load-tracking", CompatKind::kBool, "false",
     "server-load tracking"},
    {"--enable-force-include-usage", CompatKind::kBool, "false",
     "forced usage reporting"},
    {"--enable-tokenizer-info-endpoint", CompatKind::kBool, "false",
     "the tokenizer-info endpoint"},
    {"--enable-log-outputs", CompatKind::kBool, "false", "log shaping", true},
    {"--enable-log-deltas", CompatKind::kBool, "true", "log shaping", true},
    {"--tokens-only", CompatKind::kBool, "false", "tokens-only mode"},
    {"--fingerprint-mode", CompatKind::kString, "full",
     "fingerprint-mode control"},
    {"--fingerprint-value", CompatKind::kString, "",
     "fingerprint-value control"},
    {"--enable-flash-late-interaction", CompatKind::kBool, "true",
     "flash late interaction"},
    {"--data-parallel-supervisor-port", CompatKind::kInt, "9256",
     "data parallelism"},
    {"--dp-supervisor-probe-interval-s", CompatKind::kDouble, "5.0",
     "data parallelism"},
    {"--dp-supervisor-probe-timeout-s", CompatKind::kDouble, "5.0",
     "data parallelism"},
    {"--dp-supervisor-probe-failure-threshold", CompatKind::kInt, "3",
     "data parallelism"},

    // ModelConfig
    {"--tokenizer", CompatKind::kString, "", "a separate tokenizer path"},
    {"--tokenizer-mode", CompatKind::kString, "auto", "tokenizer modes"},
    {"--tokenizer-revision", CompatKind::kString, "", "tokenizer revisions"},
    {"--hf-config-path", CompatKind::kString, "", "separate HF config paths"},
    {"--config-format", CompatKind::kString, "auto", "config formats"},
    {"--runner", CompatKind::kString, "auto", "runner selection"},
    {"--convert", CompatKind::kString, "auto", "model conversion"},
    {"--seed", CompatKind::kInt, "0",
     "engine-level seeding (per-request `seed` is supported)"},
    {"--revision", CompatKind::kString, "", "Hugging Face Hub revisions"},
    {"--code-revision", CompatKind::kString, "", "Hugging Face Hub revisions"},
    {"--hf-token", CompatKind::kString, "", "Hugging Face Hub access"},
    {"--hf-overrides", CompatKind::kString, "{}", "HF config overrides"},
    {"--model-class-overrides", CompatKind::kString, "{}",
     "model-class overrides"},
    {"--model-impl", CompatKind::kString, "auto", "model-impl selection"},
    {"--allow-deprecated-quantization", CompatKind::kBool, "false",
     "deprecated quantization schemes"},
    {"--trust-remote-code", CompatKind::kBool, "false",
     "remote code (InferX loads local checkpoints only)", true},
    {"--skip-tokenizer-init", CompatKind::kBool, "false",
     "skipping tokenizer init"},
    {"--enable-prompt-embeds", CompatKind::kBool, "false",
     "prompt embeddings"},
    {"--enable-return-routed-experts", CompatKind::kBool, "false",
     "routed-expert reporting"},
    {"--allowed-local-media-path", CompatKind::kString, "",
     "multimodal inputs"},
    {"--allowed-media-domains", CompatKind::kString, "", "multimodal inputs"},
    {"--disable-sliding-window", CompatKind::kBool, "false",
     "sliding-window control"},
    {"--disable-cascade-attn", CompatKind::kBool, "true",
     "cascade attention"},
    {"--enable-sleep-mode", CompatKind::kBool, "false", "sleep mode"},
    {"--enable-cumem-allocator", CompatKind::kBool, "false",
     "the cumem allocator"},
    {"--generation-config", CompatKind::kString, "auto",
     "generation-config defaults"},
    {"--override-generation-config", CompatKind::kString, "{}",
     "generation-config overrides"},
    {"--logits-processors", CompatKind::kString, "",
     "custom logits processors"},
    {"--logprobs-mode", CompatKind::kString, "raw_logprobs",
     "alternate logprobs modes"},
    {"--use-fp64-gumbel", CompatKind::kBool, "false", "fp64 Gumbel sampling"},
    {"--pooler-config", CompatKind::kString, "", "pooling models"},
    {"--io-processor-plugin", CompatKind::kString, "",
     "IO-processor plugins"},
    {"--renderer-num-workers", CompatKind::kInt, "1", "renderer workers"},
    {"--override-attention-dtype", CompatKind::kString, "",
     "attention-dtype overrides"},

    // LoadConfig
    {"--load-format", CompatKind::kString, "auto", "alternate load formats"},
    {"--download-dir", CompatKind::kString, "",
     "Hub downloads (InferX loads local checkpoints only)"},
    {"--ignore-patterns", CompatKind::kString, "", "load ignore-patterns"},
    {"--model-loader-extra-config", CompatKind::kString, "{}",
     "model-loader extra config"},
    {"--pt-load-map-location", CompatKind::kString, "cpu",
     "PyTorch checkpoint loading"},
    {"--safetensors-load-strategy", CompatKind::kString, "",
     "safetensors load strategies"},
    {"--safetensors-prefetch-num-threads", CompatKind::kInt, "8",
     "safetensors prefetch tuning"},
    {"--safetensors-prefetch-block-size", CompatKind::kInt, "16777216",
     "safetensors prefetch tuning"},
    {"--use-tqdm-on-load", CompatKind::kBool, "true", "load progress bars",
     true},

    // ParallelConfig
    {"--pipeline-parallel-size", CompatKind::kInt, "1",
     "pipeline parallelism"},
    {"--data-parallel-size", CompatKind::kInt, "1", "data parallelism"},
    {"--data-parallel-rank", CompatKind::kInt, "", "data parallelism"},
    {"--data-parallel-start-rank", CompatKind::kInt, "", "data parallelism"},
    {"--data-parallel-size-local", CompatKind::kInt, "", "data parallelism"},
    {"--data-parallel-address", CompatKind::kString, "", "data parallelism"},
    {"--data-parallel-rpc-port", CompatKind::kInt, "", "data parallelism"},
    {"--data-parallel-backend", CompatKind::kString, "mp",
     "data parallelism"},
    {"--data-parallel-hybrid-lb", CompatKind::kBool, "false",
     "data parallelism"},
    {"--data-parallel-external-lb", CompatKind::kBool, "false",
     "data parallelism"},
    {"--data-parallel-multi-port-external-lb", CompatKind::kBool, "false",
     "data parallelism"},
    {"--prefill-context-parallel-size", CompatKind::kInt, "1",
     "context parallelism"},
    {"--decode-context-parallel-size", CompatKind::kInt, "1",
     "context parallelism"},
    {"--dcp-comm-backend", CompatKind::kString, "ag_rs",
     "context parallelism"},
    {"--dcp-kv-cache-interleave-size", CompatKind::kInt, "1",
     "context parallelism"},
    {"--cp-kv-cache-interleave-size", CompatKind::kInt, "1",
     "context parallelism"},
    {"--distributed-executor-backend", CompatKind::kString, "",
     "distributed executor backends"},
    {"--distributed-timeout-seconds", CompatKind::kInt, "",
     "distributed timeouts"},
    {"--cpu-distributed-timeout-seconds", CompatKind::kInt, "",
     "distributed timeouts"},
    {"--master-addr", CompatKind::kString, "127.0.0.1",
     "multi-node serving"},
    {"--master-port", CompatKind::kInt, "29501", "multi-node serving"},
    {"--nnodes", CompatKind::kInt, "1", "multi-node serving"},
    {"--node-rank", CompatKind::kInt, "0", "multi-node serving"},
    {"--numa-bind", CompatKind::kBool, "false", "NUMA binding"},
    {"--numa-bind-nodes", CompatKind::kString, "", "NUMA binding"},
    {"--numa-bind-cpus", CompatKind::kString, "", "NUMA binding"},
    {"--max-parallel-loading-workers", CompatKind::kInt, "",
     "parallel loading workers"},
    {"--disable-custom-all-reduce", CompatKind::kBool, "false",
     "custom all-reduce (InferX has none to disable)", true},
    {"--disable-nccl-for-dp-synchronization", CompatKind::kString, "",
     "data parallelism"},
    {"--ray-workers-use-nsight", CompatKind::kBool, "false", "Ray workers"},
    {"--enable-expert-parallel", CompatKind::kBool, "false",
     "expert parallelism"},
    {"--enable-ep-weight-filter", CompatKind::kBool, "false",
     "expert parallelism"},
    {"--enable-eplb", CompatKind::kBool, "false",
     "expert-parallel load balancing"},
    {"--eplb-config", CompatKind::kString, "",
     "expert-parallel load balancing"},
    {"--expert-placement-strategy", CompatKind::kString, "linear",
     "expert placement strategies"},
    {"--enable-elastic-ep", CompatKind::kBool, "false",
     "elastic expert parallelism"},
    {"--all2all-backend", CompatKind::kString, "allgather_reducescatter",
     "all-to-all backends"},
    {"--enable-dbo", CompatKind::kBool, "false", "dual-batch overlap"},
    {"--ubatch-size", CompatKind::kInt, "0", "dual-batch overlap"},
    {"--dbo-decode-token-threshold", CompatKind::kInt, "32",
     "dual-batch overlap"},
    {"--dbo-prefill-token-threshold", CompatKind::kInt, "512",
     "dual-batch overlap"},
    {"--worker-cls", CompatKind::kString, "auto", "custom worker classes"},
    {"--worker-extension-cls", CompatKind::kString, "",
     "custom worker classes"},

    // CacheConfig
    {"--prefix-caching-hash-algo", CompatKind::kString, "sha256",
     "prefix-cache hash selection"},
    {"--prefix-match-unit", CompatKind::kInt, "", "prefix-match units"},
    {"--kv-cache-dtype-skip-layers", CompatKind::kString, "",
     "per-layer KV dtypes"},
    {"--calculate-kv-scales", CompatKind::kBool, "false",
     "runtime KV-scale calculation"},
    {"--kv-sharing-fast-prefill", CompatKind::kBool, "false",
     "KV-sharing fast prefill"},
    {"--kv-offloading-size", CompatKind::kDouble, "", "KV offloading"},
    {"--kv-offloading-backend", CompatKind::kString, "native",
     "KV offloading"},
    {"--mamba-block-size", CompatKind::kInt, "", "Mamba models"},
    {"--mamba-cache-dtype", CompatKind::kString, "auto", "Mamba models"},
    {"--mamba-ssm-cache-dtype", CompatKind::kString, "auto", "Mamba models"},
    {"--mamba-cache-mode", CompatKind::kString, "none", "Mamba models"},

    // OffloadConfig
    {"--cpu-offload-gb", CompatKind::kDouble, "0", "CPU offloading"},
    {"--cpu-offload-params", CompatKind::kString, "", "CPU offloading"},
    {"--offload-backend", CompatKind::kString, "auto", "CPU offloading"},
    {"--offload-group-size", CompatKind::kInt, "0", "CPU offloading"},
    {"--offload-num-in-group", CompatKind::kInt, "1", "CPU offloading"},
    {"--offload-prefetch-step", CompatKind::kInt, "1", "CPU offloading"},
    {"--offload-params", CompatKind::kString, "", "CPU offloading"},

    // SchedulerConfig
    {"--max-num-scheduled-tokens", CompatKind::kInt, "",
     "a separate scheduled-token cap (use --max-num-batched-tokens)"},
    {"--max-num-partial-prefills", CompatKind::kInt, "1",
     "bounded partial prefills"},
    {"--max-long-partial-prefills", CompatKind::kInt, "1",
     "bounded partial prefills"},
    {"--long-prefill-token-threshold", CompatKind::kInt, "0",
     "long-prefill thresholds"},
    {"--disable-chunked-mm-input", CompatKind::kBool, "false",
     "multimodal inputs"},
    {"--disable-hybrid-kv-cache-manager", CompatKind::kString, "",
     "the hybrid KV-cache manager"},
    {"--scheduling-policy", CompatKind::kString, "fcfs",
     "priority scheduling"},
    {"--scheduler-cls", CompatKind::kString, "", "custom scheduler classes"},
    {"--scheduler-reserve-full-isl", CompatKind::kBool, "true",
     "reserve-full-ISL control"},
    {"--prefill-schedule-interval", CompatKind::kInt, "1",
     "prefill-schedule intervals"},
    {"--watermark", CompatKind::kDouble, "0.0", "cache watermarks"},
    {"--async-scheduling", CompatKind::kString, "", "async scheduling"},
    {"--stream-interval", CompatKind::kInt, "1", "stream intervals"},

    // MultiModalConfig
    {"--limit-mm-per-prompt", CompatKind::kString, "{}",
     "multimodal inputs"},
    {"--enable-mm-embeds", CompatKind::kBool, "false", "multimodal inputs"},
    {"--interleave-mm-strings", CompatKind::kBool, "false",
     "multimodal inputs"},
    {"--language-model-only", CompatKind::kBool, "false",
     "multimodal inputs"},
    {"--media-io-kwargs", CompatKind::kString, "{}", "multimodal inputs"},
    {"--mm-processor-kwargs", CompatKind::kString, "", "multimodal inputs"},
    {"--mm-processor-cache-gb", CompatKind::kDouble, "4",
     "multimodal inputs"},
    {"--mm-processor-cache-type", CompatKind::kString, "lru",
     "multimodal inputs"},
    {"--mm-shm-cache-max-object-size-mb", CompatKind::kInt, "128",
     "multimodal inputs"},
    {"--mm-encoder-only", CompatKind::kBool, "false", "multimodal inputs"},
    {"--mm-encoder-tp-mode", CompatKind::kString, "weights",
     "multimodal inputs"},
    {"--mm-encoder-attn-backend", CompatKind::kString, "",
     "multimodal inputs"},
    {"--mm-encoder-attn-dtype", CompatKind::kString, "",
     "multimodal inputs"},
    {"--mm-encoder-fp8-scale-path", CompatKind::kString, "",
     "multimodal inputs"},
    {"--mm-encoder-fp8-scale-save-path", CompatKind::kString, "",
     "multimodal inputs"},
    {"--mm-encoder-fp8-scale-save-margin", CompatKind::kDouble, "1.5",
     "multimodal inputs"},
    {"--mm-tensor-ipc", CompatKind::kString, "direct_rpc",
     "multimodal inputs"},
    {"--mm-ipc-gpu-memory-gb", CompatKind::kDouble, "0",
     "multimodal inputs"},
    {"--skip-mm-profiling", CompatKind::kBool, "false",
     "multimodal inputs"},
    {"--video-pruning-rate", CompatKind::kDouble, "", "multimodal inputs"},

    // LoRAConfig
    {"--enable-lora", CompatKind::kBool, "false", "LoRA adapters"},
    {"--max-loras", CompatKind::kInt, "1", "LoRA adapters"},
    {"--max-lora-rank", CompatKind::kInt, "16", "LoRA adapters"},
    {"--max-cpu-loras", CompatKind::kInt, "", "LoRA adapters"},
    {"--lora-dtype", CompatKind::kString, "auto", "LoRA adapters"},
    {"--lora-target-modules", CompatKind::kString, "", "LoRA adapters"},
    {"--fully-sharded-loras", CompatKind::kBool, "false", "LoRA adapters"},
    {"--default-mm-loras", CompatKind::kString, "", "LoRA adapters"},
    {"--specialize-active-lora", CompatKind::kBool, "false",
     "LoRA adapters"},
    {"--enable-tower-connector-lora", CompatKind::kBool, "false",
     "LoRA adapters"},
    {"--enable-mixed-moe-lora-format", CompatKind::kBool, "false",
     "LoRA adapters"},
    {"--enable-moe-shared-loras", CompatKind::kBool, "false",
     "LoRA adapters"},

    // StructuredOutputs / reasoning
    {"--reasoning-parser", CompatKind::kString, "", "reasoning parsing"},
    {"--reasoning-parser-plugin", CompatKind::kString, "",
     "reasoning parsing"},
    {"--structured-outputs-config", CompatKind::kString, "",
     "structured outputs"},
    {"--reasoning-config", CompatKind::kString, "", "reasoning parsing"},

    // Speculative decoding
    {"--speculative-config", CompatKind::kString, "",
     "speculative decoding"},
    {"--spec-method", CompatKind::kString, "", "speculative decoding"},
    {"--spec-model", CompatKind::kString, "", "speculative decoding"},
    {"--spec-tokens", CompatKind::kString, "", "speculative decoding"},

    // ObservabilityConfig
    {"--otlp-traces-endpoint", CompatKind::kString, "",
     "OpenTelemetry tracing"},
    {"--collect-detailed-traces", CompatKind::kString, "",
     "OpenTelemetry tracing"},
    {"--show-hidden-metrics-for-version", CompatKind::kString, "",
     "hidden-metric compatibility"},
    {"--kv-cache-metrics", CompatKind::kBool, "false", "KV-cache metrics"},
    {"--kv-cache-metrics-sample", CompatKind::kDouble, "0.01",
     "KV-cache metrics"},
    {"--cudagraph-metrics", CompatKind::kBool, "false",
     "CUDA-graph metrics"},
    {"--enable-layerwise-nvtx-tracing", CompatKind::kBool, "false",
     "layerwise NVTX tracing"},
    {"--enable-mfu-metrics", CompatKind::kBool, "false", "MFU metrics"},
    {"--enable-logging-iteration-details", CompatKind::kBool, "false",
     "log shaping", true},
    {"--jit-monitor-mode", CompatKind::kString, "warn", "JIT monitoring"},
    {"--jit-monitor-verbose", CompatKind::kBool, "false", "JIT monitoring"},

    // Attention / Mamba / Kernel / Compilation config groups
    {"--attention-backend", CompatKind::kString, "",
     "attention-backend selection"},
    {"--attention-config", CompatKind::kString, "",
     "attention-backend selection"},
    {"--mamba-backend", CompatKind::kString, "triton", "Mamba models"},
    {"--enable-mamba-cache-stochastic-rounding", CompatKind::kBool, "false",
     "Mamba models"},
    {"--mamba-cache-philox-rounds", CompatKind::kInt, "0", "Mamba models"},
    {"--moe-backend", CompatKind::kString, "auto",
     "MoE-backend selection"},
    {"--linear-backend", CompatKind::kString, "auto",
     "linear-backend selection"},
    {"--enable-flashinfer-autotune", CompatKind::kString, "",
     "FlashInfer autotuning"},
    {"--enable-cutedsl-warmup", CompatKind::kBool, "true",
     "CuteDSL warmup"},
    {"--enable-bf16x3-router-gemm", CompatKind::kString, "",
     "router-GEMM selection"},
    {"--ir-op-priority", CompatKind::kString, "", "IR op priorities"},
    {"--kernel-config", CompatKind::kString, "", "kernel-config overrides"},
    {"--cudagraph-capture-sizes", CompatKind::kString, "",
     "explicit CUDA-graph capture sizes"},
    {"--max-cudagraph-capture-size", CompatKind::kInt, "",
     "explicit CUDA-graph capture sizes"},
    {"--compilation-config", CompatKind::kString, "", "torch.compile"},

    // VllmConfig top-level JSON groups
    {"--additional-config", CompatKind::kString, "{}",
     "additional-config passthrough"},
    {"--diffusion-config", CompatKind::kString, "", "diffusion models"},
    {"--ec-transfer-config", CompatKind::kString, "",
     "encoder-cache transfer"},
    {"--kv-events-config", CompatKind::kString, "", "KV event publishing"},
    {"--kv-transfer-config", CompatKind::kString, "",
     "KV transfer (disaggregated serving)"},
    {"--optimization-level", CompatKind::kString, "2",
     "optimization-level control"},
    {"--performance-mode", CompatKind::kString, "balanced",
     "performance modes"},
    {"--profiler-config", CompatKind::kString, "", "the torch profiler"},
    {"--weight-transfer-config", CompatKind::kString, "",
     "weight transfer"},
};

// The frontend rows above, reused verbatim for inferx-gateway, which has no
// engine behind it. The gateway registers its own real --chat-template, so
// that row is excluded.
constexpr std::string_view kGatewayStubNames[] = {
    "--uds", "--root-path", "--middleware", "--allow-credentials",
    "--allowed-origins", "--allowed-methods", "--allowed-headers",
    "--ssl-keyfile", "--ssl-certfile", "--ssl-ca-certs", "--ssl-cert-reqs",
    "--ssl-ciphers", "--enable-ssl-refresh", "--uvicorn-log-level",
    "--disable-uvicorn-access-log", "--disable-access-log-for-endpoints",
    "--log-config-file", "--max-log-len", "--log-error-stack",
    "--disable-fastapi-docs", "--enable-offline-docs",
    "--h11-max-incomplete-event-size", "--h11-max-header-count",
    "--enable-request-id-headers", "--chat-template-content-format",
    "--trust-request-chat-template", "--default-chat-template-kwargs",
    "--response-role", "--return-tokens-as-token-ids", "--lora-modules",
    "--enable-auto-tool-choice", "--exclude-tools-when-tool-choice-none",
    "--tool-call-parser", "--tool-parser-plugin", "--tool-server",
    "--enable-prompt-tokens-details", "--enable-per-request-metrics",
    "--enable-server-load-tracking", "--enable-force-include-usage",
    "--enable-tokenizer-info-endpoint", "--enable-log-outputs",
    "--enable-log-deltas", "--tokens-only", "--fingerprint-mode",
    "--fingerprint-value", "--enable-flash-late-interaction",
};

std::vector<CompatStub> BuildGatewayStubs() {
  std::vector<CompatStub> stubs;
  for (std::string_view name : kGatewayStubNames) {
    for (const CompatStub& stub : kServeStubs) {
      if (stub.name == name) {
        stubs.push_back(stub);
        break;
      }
    }
  }
  return stubs;
}

// "--enable-lora" -> "--enable-lora,!--no-enable-lora": vLLM registers every
// boolean with its negated twin, so scripts pass either spelling.
std::string BoolFlagSpec(std::string_view name) {
  return absl::StrCat(name, ",!--no-", name.substr(2));
}

bool ParsesAsTrue(std::string_view text) {
  return text == "true" || text == "1" || text == "True";
}

}  // namespace

std::span<const CompatStub> ServeCompatStubs() { return kServeStubs; }

std::span<const CompatStub> GatewayCompatStubs() {
  static const std::vector<CompatStub>* stubs =
      new std::vector<CompatStub>(BuildGatewayStubs());
  return *stubs;
}

void AddCompatStubs(CLI::App& app, std::span<const CompatStub> stubs,
                    CompatState& state) {
  for (const CompatStub& stub : stubs) {
    CLI::Option* option = nullptr;
    if (stub.kind == CompatKind::kBool) {
      state.bools.push_back(ParsesAsTrue(stub.default_text));
      option = app.add_flag(BoolFlagSpec(stub.name), state.bools.back(),
                            std::string(stub.feature));
    } else {
      state.values.emplace_back();
      option = app.add_option(std::string(stub.name), state.values.back(),
                              std::string(stub.feature));
    }
    option->group(std::string(kGroup));
    state.bound.emplace_back(&stub, option);
  }
}

Status CheckCompatStubs(std::span<const CompatStub> stubs,
                        const CompatState& state) {
  (void)stubs;
  size_t value_index = 0;
  size_t bool_index = 0;
  for (const auto& [stub, option] : state.bound) {
    const bool is_bool = stub->kind == CompatKind::kBool;
    const std::string given = is_bool
                                  ? (state.bools[bool_index] ? "true" : "false")
                                  : state.values[value_index];
    if (is_bool) {
      ++bool_index;
    } else {
      ++value_index;
    }
    if (option->count() == 0) continue;

    if (stub->accept_any) {
      LOG(INFO) << stub->name << "=" << given
                << " accepted for vLLM compatibility; it has no effect on "
                   "InferX";
      continue;
    }

    bool matches_default = false;
    if (!stub->default_text.empty()) {
      switch (stub->kind) {
        case CompatKind::kBool:
          matches_default =
              (given == "true") == ParsesAsTrue(stub->default_text);
          break;
        case CompatKind::kInt: {
          char* end = nullptr;
          const long long given_value = std::strtoll(given.c_str(), &end, 10);
          const bool given_ok = end != given.c_str() && *end == '\0';
          matches_default =
              given_ok &&
              given_value ==
                  std::strtoll(std::string(stub->default_text).c_str(),
                               nullptr, 10);
          break;
        }
        case CompatKind::kDouble: {
          char* end = nullptr;
          const double given_value = std::strtod(given.c_str(), &end);
          const bool given_ok = end != given.c_str() && *end == '\0';
          matches_default =
              given_ok &&
              given_value == std::strtod(std::string(stub->default_text).c_str(),
                                         nullptr);
          break;
        }
        case CompatKind::kString:
          matches_default = given == stub->default_text;
          break;
      }
    }
    if (matches_default) continue;

    if (stub->default_text.empty()) {
      return InvalidArgumentError(
          absl::StrCat(stub->name, "=", given, " requests ", stub->feature,
                       ", which is not supported by InferX"));
    }
    return InvalidArgumentError(absl::StrCat(
        stub->name, "=", given, " requests ", stub->feature,
        ", which is not supported by InferX (only the vLLM default '",
        stub->default_text, "' is accepted)"));
  }
  return OkStatus();
}

std::string Sha256Hex(std::string_view data) {
  std::array<unsigned char, EVP_MAX_MD_SIZE> digest{};
  unsigned int digest_size = 0;
  if (EVP_Digest(data.data(), data.size(), digest.data(), &digest_size,
                 EVP_sha256(), nullptr) != 1) {
    return {};
  }
  std::ostringstream encoded;
  encoded << std::hex << std::setfill('0');
  for (unsigned int i = 0; i < digest_size; ++i) {
    encoded << std::setw(2) << static_cast<unsigned int>(digest[i]);
  }
  return encoded.str();
}

}  // namespace inferx
