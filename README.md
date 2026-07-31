# InferX

A high-performance LLM inference server in C++20, aiming at throughput and
latency comparable to vLLM and SGLang — without Python or libtorch in the
serving path.

**Status: M0.** The build system and core memory layer exist and are tested.
There is no model, no kernel, and no server yet.

## Why another engine

Python inference engines pay a structural tax that C++ does not, and InferX is
shaped to collect the refund rather than to imitate their architecture:

- **No GIL, so no process-per-rank.** vLLM and SGLang run one OS process per
  tensor-parallel rank and serialize each step's batch metadata between them.
  InferX runs one process with one thread per rank, sharing the scheduler's
  output by pointer — no IPC, no serialization, no multi-process restart dance.
- **No interpreter, so the CPU can run far ahead.** Per-step scheduling costs
  microseconds instead of milliseconds, which keeps the GPU fed at large batch
  sizes.
- **Real ownership semantics.** KV blocks, request state, and weight buffers get
  RAII lifetimes and single-owner rules checkable at compile time.

## Design

[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) is the design of record —
components, data flow, execution model, the decision register, and the risks.
Start there.

The load-bearing split: **the scheduler holds all the policy and never touches
CUDA; the executor touches all the CUDA and holds no policy.** That is what makes
continuous batching, the paged KV manager, and the radix prefix cache testable on
a machine with no GPU.

## Getting started

```bash
git clone --recurse-submodules <url> && cd inferx
./scripts/install-cuda.sh     # needs sudo; CUDA 13.x is required
./scripts/bootstrap.sh        # submodules, configure, build, test
```

No GPU, or not upgraded yet:

```bash
./scripts/bootstrap.sh --host-only
```

See [`docs/DEVELOPMENT.md`](docs/DEVELOPMENT.md) for toolchain details,
conventions, and troubleshooting.

## Roadmap

| | Milestone | Status |
|---|---|---|
| M0 | Toolchain, build system, core memory layer | **done** |
| M1 | CUTLASS W4A16 GEMM vs cuBLASLt baseline | next |
| M2 | Safetensors loader + Llama forward pass | |
| M3 | Paged KV + FlashInfer attention + naive scheduler | |
| M4 | HTTP server, tokenizer, OpenAI-compatible streaming | |
| M5 | Continuous batching, chunked prefill, prefix cache | |
| M6 | Overlap pipeline + CUDA graphs | |
| M7 | Tensor parallelism | |
| M8 | W4A16 weights + FP8 KV cache | |
| M9 | MoE and MLA | |
| M10 | Benchmarks vs vLLM/SGLang | |

M1 deliberately precedes the model layer: torch-free quantized GEMM is the
largest unknown in the design, and finding out it is intractable after building
everything on top of it would be expensive.

## Target hardware

Primary development target is a single sm_89 (Ada) GPU with 16 GB, which is why
quantization is v1 scope rather than a later optimization. Tensor parallelism is
designed in from the start but cannot be performance-tested on that box; see
ARCHITECTURE.md §7.4 for how correctness is validated without multi-GPU hardware.
