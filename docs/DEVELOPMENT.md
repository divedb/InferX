# Development setup

## Toolchain requirements

| Component | Required | Notes |
|---|---|---|
| CUDA toolkit | **13.0+** | Enforced at configure time. See below. |
| NVIDIA driver | r580+ | Reports "CUDA Version: 13.x" in `nvidia-smi`. On WSL2 this is the **Windows** driver. |
| C++ compiler | GCC 13+ or Clang 18+ | C++20. GCC 13 lacks `std::expected`, hence `absl::StatusOr`. |
| CMake | 3.28+ | 3.31+ recommended with CUDA 13.x. |
| GPU | sm_75+ | CUDA 13 dropped Maxwell/Pascal/Volta. Primary target is sm_89 (Ada). |
| NCCL | optional | Required for `--tensor-parallel-size 2`; TP=1 and HostSim build without it. |

For the first production TP runtime, install NCCL's development package so
both `nccl.h` and `libnccl.so` are discoverable at configure time. CMake prints
`NCCL backend : ON` when it is enabled. A two-GPU launch is:

```bash
./build-cuda/src/server/inferx-serve \
  --model /path/to/checkpoint \
  --tensor-parallel-size 2 \
  --devices 0,1 \
  --comm-backend nccl
```

Both GPUs must be in the same machine and visible to the process. Record
`nvidia-smi topo -m` and verify CUDA P2P access before interpreting TP
performance. The complete rental-host acceptance procedure is in
[TP_VALIDATION.md](TP_VALIDATION.md).

### API-key authentication

Authentication is optional for local development. Production launches should
pass one or more SHA-256 API-key digests; raw keys are never stored in server
configuration:

```bash
key_hash="$(printf %s "$INFERX_API_KEY" | sha256sum | cut -d' ' -f1)"
./build/src/server/inferx-serve \
  --model /path/to/checkpoint \
  --api-key-sha256 "$key_hash"
```

Clients send `Authorization: Bearer <key>`. Health endpoints remain public for
orchestrator probes. Models, completions, metrics, and legacy stats require a
valid key whenever at least one hash is configured.

### Why CUDA 13.x

The M0 core only uses CUDA runtime calls that are stable across 12.x and 13.x,
but the kernel work starting at M1 does not have that luxury:

- **CUTLASS 4.x** is required for the FP8 and grouped-GEMM paths that MoE and
  W4A16 quantization depend on. CUTLASS 3.x is CUDA 12.x only.
- **FlashInfer's MLA attention wrapper** — needed for DeepSeek-V2/V3 support —
  only exists in versions that assume a recent toolkit.
- CUDA 12.0–12.3 **cannot compile against libstdc++ 13 headers** at all, so on a
  current Ubuntu the older toolkit is a dead end regardless.

This is risk R1 in [ARCHITECTURE.md](ARCHITECTURE.md), and retiring it is the
first deliverable of M0.

### Installing

```bash
./scripts/install-cuda.sh          # needs sudo; installs cuda-toolkit-13-0
```

Then put the toolkit ahead of anything else on your `PATH`:

```bash
export CUDA_HOME=/usr/local/cuda-13.0
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}
```

> **The most common failure.** Ubuntu's own `nvidia-cuda-toolkit` package puts
> `nvcc` in `/usr/bin`, which wins on the default `PATH` even after you install
> 13.x under `/usr/local`. If `nvcc --version` still says 12.x, run
> `sudo apt-get remove -y nvidia-cuda-toolkit`.

**On WSL2**, install the `wsl-ubuntu` repo variant (the script picks it
automatically). It deliberately ships no display driver — the GPU driver comes
from the Windows host through `/usr/lib/wsl/lib`, and installing a Linux driver
on top of it breaks CUDA. Driver updates happen in Windows, not here.

## Building

```bash
./scripts/bootstrap.sh              # submodules, configure, build, test
```

Or by hand:

```bash
git submodule update --init --recursive
cmake -S . -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build -j$(nproc)
```

### Build configurations

| Command | Use |
|---|---|
| `./scripts/bootstrap.sh` | Normal development build |
| `./scripts/bootstrap.sh --debug` | Assertions, no optimization |
| `./scripts/bootstrap.sh --asan` | AddressSanitizer + UBSan |
| `./scripts/bootstrap.sh --host-only` | **No CUDA at all** |

`--host-only` is not a fallback for a broken CUDA install — it is a first-class
configuration. The scheduler, KV block manager, prefix cache, and the entire
memory layer are required to build and test without a GPU, because that is what
keeps their logic separable from device concerns. See ARCHITECTURE.md §3.1.

### Asynchronous HTTP dependencies

The Beast/Folly HTTP foundation is enabled by default. Boost is pinned to
1.91.0, Folly to `v2026.08.03.00`, and fast_float to 8.0.0. Boost and Folly
use `update = none` in `.gitmodules`, so initialize their source trees
explicitly after cloning. The pinned fast_float is required because Ubuntu
24.04's 3.9 package is too old for this Folly revision.

On Ubuntu 24.04, install Folly's required development packages first:

```bash
sudo apt-get update
sudo apt-get install -y \
  libfmt-dev \
  libgoogle-glog-dev \
  libgflags-dev \
  libevent-dev \
  libssl-dev
```

Initialize them once, then configure normally:

```bash
git submodule update --init --checkout \
  third_party/boost third_party/folly third_party/fast_float
git -C third_party/boost submodule update --init --depth 1
cmake -S . -B build
```

Folly's platform dependencies must be installed before configuring.
First-party code consumes the dependencies through the narrow `inferx::beast`
and `inferx::folly_coro` targets; do not link upstream targets directly. See
`http_server.md` for the transport decision and migration plan.

### Which CUDA gets used

Two things resolve independently: `find_package(CUDAToolkit)` finds headers and
libraries, and `enable_language(CUDA)` finds the compiler on `PATH`. They can
disagree. A distro `nvidia-cuda-toolkit` puts nvcc at `/usr/bin/nvcc`, which
shadows `/usr/local/cuda/bin/nvcc` — so a build can validate one toolkit and
then compile every `.cu` with a different, older one.

The build therefore points both at `/usr/local/cuda` unless you say otherwise,
and **fails hard if they end up mismatched anyway**. To use a different toolkit:

```bash
cmake -S . -B build -DINFERX_CUDA_ROOT_HINT=/usr/local/cuda-13.0
```

An explicit `-DCMAKE_CUDA_COMPILER`, `-DCUDAToolkit_ROOT`, or the `CUDACXX` /
`CUDAToolkit_ROOT` environment variables all take precedence over the default,
which is what makes a second toolkit testable. The configure summary prints both
resolved paths — if they ever disagree, that is a bug, and the build says so
rather than picking one silently:

```
--   CUDA toolkit    : 13.0.88 (/usr/local/cuda/bin)
--   CUDA compiler   : 13.0.88 (/usr/local/cuda/bin/nvcc)
```

The version floor is enforced against the **compiler**, not the toolkit, because
the compiler is what actually has to swallow our headers.

### The unsupported-CUDA escape hatch

```bash
cmake -S . -B build -DINFERX_ALLOW_UNSUPPORTED_CUDA=ON
```

Configures against a pre-13.0 toolkit with a loud warning. This exists so the
runtime-API layer can be compile-checked on a machine that has not been upgraded
yet. Do not use it for anything you intend to measure.

Below the floor, **every target containing device code is skipped** —
`csrc` and the kernel tests with it. The configure summary says so
(`Device code : SKIPPED`). The reason is concrete rather than precautionary:
nvcc 12.x cannot evaluate `std::source_location::current()`, which
`absl/status/status.h` uses unconditionally, so any `.cu` that includes an
inferx core header fails in the frontend. A green build under this flag
therefore proves less than it looks like it does — it has not compiled a single
line of device code.

## Layout

```
include/inferx/     Public headers, mirroring src/
src/core/           DType, Tensor, Status, arenas, allocators
csrc/               .cu — device code. Needs CUDA >= 13.0 to build at all
cmake/              Options, CUDA config, dependency wiring
scripts/            Setup and bootstrap
third_party/        Submodules, pinned
docs/               ARCHITECTURE.md is the design of record
```

## Conventions

**Errors.** `absl::Status` / `absl::StatusOr` everywhere, via
`inferx/core/status.h`. Use `INFERX_RETURN_IF_ERROR` and
`INFERX_ASSIGN_OR_RETURN`. Exceptions are allowed only for unrecoverable startup
faults, and must never cross a thread boundary or appear in a per-step path.

**CUDA source files.** Only files containing `__global__` kernels are `.cu`.
Anything that merely *calls* the CUDA runtime API includes
`<cuda_runtime_api.h>` from an ordinary `.cc` — it is a C header and the host
compiler handles it. This keeps nvcc out of most of the build and is why the
device memory layer is testable without a device compiler.

**Device support is compile-time and run-time.** `kCudaEnabled` says the binary
has device support; `CudaAvailable()` says a GPU is actually usable right now.
GPU-dependent tests gate on the latter with `GTEST_SKIP()`.

**Sizes are in bits.** `DTypeBits()`, not a byte size, because int4 is a real
weight format. Compute byte counts with `DTypeByteSize()`, which rounds up.

**Two tensor types, and the choice matters.** `Tensor` is the owning handle
(refcounted, shares `Storage`, one pointer wide). `TensorView` is the
non-owning POD. Establish ownership once — at weight load or request admission —
as a `Tensor`, then pass `TensorView` downstream. A `Tensor` cannot be passed to
a kernel at all: `__global__` parameters must be trivially copyable. Putting
owning handles on the decode path also means an atomic increment per copy and a
heap allocation per intermediate, neither of which is affordable there.

**Refcounting: always `ThreadSafeRefCountedBase`.** `intrusive_ref_cnt_ptr.h`
offers both it and the non-atomic `RefCountedBase`, which are equally easy to
type. Anything reachable from more than one thread — which is everything in the
ownership layer, since handles cross the scheduler/executor boundary — must use
the thread-safe base. Picking the other one is a data race with no symptom until
production.

**`Shape` has no maximum rank.** `Shape::kDefaultRank` is inline capacity — how
many dimensions fit without allocating — not a limit. The engine's actual limit
is `TensorView::kMaxRank`, which exists because a `__global__` parameter must be
trivially copyable, and it is enforced in `ValidateTensorLayout`. Don't add rank
caps to `Shape`; it is meant to stay generic across model layouts.

**Borrowed storage is normal, not a hack.** `Tensor::FromBlob` and
`Storage::Borrow` wrap memory owned by an arena or an mmap'd weight file and
free nothing. Loading a 7B model is one allocation with hundreds of tensors over
it, not hundreds of allocations.

**No allocation on the step path.** The steady-state decode loop must contain no
`cudaMalloc`, no `cudaFree`, and no device synchronization — every one of them
implicitly synchronizes and collapses the overlap pipeline. Carve memory from a
`BumpArena` (weights) or `CachingAllocator` (workspace) instead.

## Submodules

Pinned deliberately; nothing tracks a moving branch.

| Path | Pin | Added at |
|---|---|---|
| `third_party/abseil-cpp` | `20260526.0` | M0 |
| `third_party/googletest` | `v1.17.0` | M0 |

### Vendored source

`include/inferx/core/intrusive_ref_cnt_ptr.h` is LLVM's
`llvm/ADT/IntrusiveRefCntPtr.h`, copied in and reparented into
`namespace inferx`. It is **Apache-2.0 WITH LLVM-exception**, not our licence:
keep the file's original header comment intact, and record it in the project's
notices before any distribution. Prefer patching it minimally so future upstream
fixes remain easy to merge.

Dependencies for later milestones, and the commands to add them, are recorded at
the bottom of `cmake/InferXDependencies.cmake`. They are added at the milestone
that needs them so an early checkout stays fast.

## Troubleshooting

**`InferX requires CUDA >= 13.0, found 12.x`** — expected on a stock Ubuntu. See
"Installing" above; check for the shadowing `/usr/bin/nvcc`.

**`Submodule third_party/... is missing or empty`** — run
`git submodule update --init --recursive`.

**`nvcc` fails inside `libstdc++` headers** — a pre-12.4 toolkit with GCC 13.
Either install CUDA 13.x or pass `-DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-12`.

**`nvidia-smi` works but `CudaDeviceCount()` returns 0** — the CUDA runtime
cannot find a driver. On WSL2, confirm `/usr/lib/wsl/lib` is on the loader path
and that no Linux NVIDIA driver package is installed.

**ASan build fails in `hash_policy_traits.h` with "is not a constant
expression"** — a GCC 13 bug compiling Abseil's hash containers under
`-fsanitize`. Use Clang; `./scripts/bootstrap.sh --asan` selects it
automatically.

**UBSan reports "load of null pointer" inside `raw_hash_set.h`** — this means
sanitizer flags were applied to first-party targets but not to Abseil. Abseil
enables extra generation-tracking state under sanitizers, so a partial
application is an ODR violation. Sanitizer flags are set globally in
`InferXOptions.cmake` for exactly this reason; do not move them onto
`inferx_flags`.
