# Deploying InferX + DeepSeek-V2-Lite to a rented GPU box

A record of the first real deployment (seetacloud/AutoDL, 2026-08-08/09),
written so the next one is a checklist instead of a debugging session. The
validation *procedure* lives in `docs/DSV2_VALIDATION.md`; this file is the
operational half: getting the machine into a state where that procedure runs,
and every failure met along the way.

## 1. Target environment

What the rented box (RTX 4090 48 GB, Ubuntu 22.04, seetacloud "westb") shipped
with, against what InferX needs:

| Component | Image shipped | Required | Action taken |
|---|---|---|---|
| GPU | RTX 4090 48 GB (sm_89), driver 580.105 | ≥ 48 GB for bf16 V2-Lite; driver r580+ | none — matches `INFERX_CUDA_ARCHS=89` default |
| CUDA toolkit | 12.8 (`/usr/local/cuda`, **not on login PATH** — check before concluding it's absent) | **13.0+** (configure-time floor) | installed `cuda-toolkit-13-0` → `/usr/local/cuda-13.0` |
| GCC | 11.4 | **13+** | `ppa:ubuntu-toolchain-r/test` → `gcc-13`/`g++-13` |
| CMake | 3.22 | **3.28+** | `pip install cmake==3.31.6` (into miniconda base; **do not** take 4.x — it removes compat with old third-party min-version declarations) |
| Python | none on PATH; miniconda at `/root/miniconda3` | any (for tooling/venvs) | used miniconda's 3.12 |
| Rust/cargo | none | required (tokenizers-cpp builds a Rust staticlib) | rustup via rsproxy.cn mirror |
| Folly system deps | none | libfmt, glog, gflags, libevent, libssl (see `docs/install.md`) | apt |
| Disk | `/` 30 GB (20 free) — too small for checkpoints | ~70 GB for two checkpoints + build | everything under `/root/autodl-tmp` (150 GB data disk) |
| RAM / CPU | 754 GB / 208 cores | — | enables **fp32 CPU golden logits** (63 GB model in RAM) |

## 2. Local prerequisites

Password auth without sshpass (not installed locally, no sudo): OpenSSH's
askpass hook, then immediately switch to key auth so every later command is
non-interactive:

```bash
printf '#!/bin/sh\necho %s\n' "'<password>'" > askpass.sh && chmod 700 askpass.sh
SSH_ASKPASS=./askpass.sh SSH_ASKPASS_REQUIRE=force \
  ssh -p <port> -o StrictHostKeyChecking=accept-new -o PubkeyAuthentication=no \
  root@<host> 'mkdir -p ~/.ssh && cat >> ~/.ssh/authorized_keys' < ~/.ssh/id_ed25519.pub
```

## 3. Syncing the project

The repo has no git remote; rsync carries source, `.git`, and the checked-out
submodules (~2 GB), so `bootstrap.sh`'s submodule sync is a no-op remotely:

```bash
rsync -az --info=progress2 -e "ssh -p <port>" \
  --exclude '/build/' --exclude '/build-*/' --exclude '/.venv-*/' \
  --exclude '/testdata/*.bin' --exclude '/bench-results/' --exclude '/.claude/' \
  ./ root@<host>:/root/autodl-tmp/InferX/
```

**Pitfall (cost one debugging round): the leading `/` on the excludes is
load-bearing.** An unanchored `--exclude 'build/'` matches *any* directory
named `build`, which silently strips `third_party/folly/build/fbcode_builder/`
— folly's own CMake modules — and configure then fails with
`include could not find requested file: FBBuildOptions`. Anchor every exclude
to the transfer root.

After syncing, `git config --global --add safe.directory '*'` on the remote
(root owns the copy; git otherwise refuses it).

## 4. Remote environment setup

In dependency order; each step's failure mode is in §8.

```bash
# CMake + tooling into miniconda base
/root/miniconda3/bin/pip install "cmake==3.31.6" ninja modelscope

# GCC 13
add-apt-repository -y ppa:ubuntu-toolchain-r/test
apt-get install -y g++-13 gcc-13

# CUDA 13 — the image had cuda-keyring installed but the repo LIST stripped,
# so `apt install cuda-toolkit-13-0` fails with "unable to locate" until:
echo "deb [signed-by=/usr/share/keyrings/cuda-archive-keyring.gpg] \
  https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/ /" \
  > /etc/apt/sources.list.d/cuda.list
apt-get update && apt-get install -y cuda-toolkit-13-0

# Folly platform deps (docs/install.md list, minus optional grpc/protobuf)
apt-get install -y libfmt-dev libgoogle-glog-dev libgflags-dev \
  libevent-dev libssl-dev libdouble-conversion-dev

# Rust for tokenizers-cpp — rsproxy.cn works well from CN networks
export RUSTUP_DIST_SERVER=https://rsproxy.cn RUSTUP_UPDATE_ROOT=https://rsproxy.cn/rustup
curl -sSf https://rsproxy.cn/rustup-init.sh | sh -s -- -y --profile minimal
# + [source.crates-io] replace-with rsproxy-sparse in ~/.cargo/config.toml

# Dedicated vLLM venv (also supplies torch/transformers for goldens)
/root/miniconda3/bin/python -m venv /root/autodl-tmp/InferX/.venv-vllm
. /root/autodl-tmp/InferX/.venv-vllm/bin/activate
pip install vllm accelerate -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### Checkpoints

hf-mirror.com sustained only ~6 MB/s for the large shards; **ModelScope
(domestic CDN) was ~17 MB/s** and hosts both models under the same ids:

```bash
modelscope download --model deepseek-ai/DeepSeek-V2-Lite      --local_dir /root/autodl-tmp/ckpt/dsv2-lite
modelscope download --model deepseek-ai/DeepSeek-V2-Lite-Chat --local_dir /root/autodl-tmp/ckpt/dsv2-lite-chat
```

Two traps: current `huggingface_hub` renamed the CLI — `huggingface-cli
download` **prints help and exits 0** (downloads nothing; the command is `hf
download` now). And ModelScope **preallocates** `.incomplete` shard files at
full size, so directory size is useless as a progress signal — watch for the
`.incomplete` suffixes to disappear instead.

## 5. Build

```bash
cd /root/autodl-tmp/InferX
export PATH=$HOME/.cargo/bin:/usr/local/cuda-13.0/bin:/root/miniconda3/bin:$PATH
cmake -S . -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DCMAKE_C_COMPILER=gcc-13 -DCMAKE_CXX_COMPILER=g++-13 \
  -DCMAKE_CUDA_HOST_COMPILER=g++-13 \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.0/bin/nvcc \
  -DINFERX_CUDA_ROOT_HINT=/usr/local/cuda-13.0

# Fast path first: the logits gate needs only kernels+model (~minutes)
cmake --build build -j64 --target deepseek_v2_reference_test \
  deepseek_v2_model_test mla_test moe_test
(cd build && ctest -L kernel)          # 28/28 passed on the 4090

# Full server (Boost + Folly, the long compile)
cmake --build build -j64 --target inferx-serve
```

**Configure order matters for cargo:** tokenizers-cpp resolves the cargo path
at *configure* time. Configuring before rustup was installed bakes
`CARGO_EXECUTABLE-NOTFOUND` into the generated build rule, which then fails as
a bare `no such file or directory`. Re-run `cmake -S . -B build` (cheap;
artifacts survive) after installing Rust.

## 6. Validation and serving

Per `docs/DSV2_VALIDATION.md`, with session amendments:

```bash
. .venv-vllm/bin/activate
python scripts/gen_deepseek_logits.py /root/autodl-tmp/ckpt/dsv2-lite \
    testdata/deepseek_v2_lite_logits.bin          # bf16 GPU reference
export INFERX_TEST_DEEPSEEK_CHECKPOINT=/root/autodl-tmp/ckpt/dsv2-lite
(cd build && ctest -R DeepseekV2Reference --output-on-failure)                      # convention A
(cd build && INFERX_DSV2_ROPE_DEINTERLEAVE=1 ctest -R DeepseekV2Reference --output-on-failure)  # convention B

INFERX_DSV2_ROPE_DEINTERLEAVE=1 ./build/src/server/inferx-serve \
  --model /root/autodl-tmp/ckpt/dsv2-lite-chat \
  --served-model-name deepseek-v2-lite-chat \
  --port 8080 --kv-blocks 8192 --block-size 16
```

Findings that changed the procedure:

- **`trust_remote_code` is dead for this checkpoint.** Its bundled
  `modeling_deepseek.py` imports `is_torch_fx_available`, removed from current
  transformers. transformers ≥ 5.x has *native* `DeepseekV2ForCausalLM`;
  `gen_deepseek_logits.py` now loads with `trust_remote_code=False`.
- **The default 5-token golden prompt cannot discriminate the RoPE
  convention** — both toggle settings passed it. A ~168-token prompt separates
  them decisively: half-split scored 131/168 argmax mismatches (deviations to
  48 % of span — systematically wrong); **deinterleaved scored 30/168 with
  mismatches scattered from position 1** — the signature of MoE
  routing-margin noise (top-6-of-64 at bf16; the gpt-oss precedent), not of
  positional rope drift. HF's own YaRN table and `attention_scaling = 1.0`
  were dumped from `transformers.modeling_rope_utils` and match our
  `ComputeYarnInvFreq` output to every printed digit, eliminating frequencies
  as a suspect. An fp32-reference rerun (the box's 754 GB RAM makes it
  practical) was in flight when this file was written, to bound the
  bf16-noise contribution before hardcoding the deinterleaved convention and
  deleting the toggle.
- Generate long-prompt goldens with
  `gen_deepseek_logits.py <ckpt> out.bin --prompt "<~150 words>"` and point
  `INFERX_TEST_DEEPSEEK_LOGITS` at the file.

## 7. vLLM baseline

`scripts/run_vllm.sh` sources `$ROOT/.venv-vllm` (repo-relative). Run with the
same checkpoint and port conventions as `bench/serve_bench.py` expects, one
engine at a time — both size their KV pool to fill the card.

## 8. Known issues and troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `include could not find requested file: FBBuildOptions` at configure | rsync stripped `third_party/*/build/` (unanchored exclude) | anchor excludes: `--exclude '/build/'` |
| `Could NOT find libevent` at configure | folly system deps missing | §4 apt list |
| `libtokenizers_c.a … no such file or directory` | no cargo, or cargo installed *after* configure (`CARGO_EXECUTABLE-NOTFOUND` baked into build.make) | install rustup, then **re-run configure** |
| `Unable to locate package cuda-toolkit-13-0` | image ships cuda-keyring but strips the repo list | write `/etc/apt/sources.list.d/cuda.list` (§4) |
| `nvcc` "missing" | CUDA 12.8 exists but isn't on the login PATH | check `/usr/local/cuda*/bin` before installing anything |
| `huggingface-cli download` exits instantly, downloads nothing | CLI renamed to `hf` in huggingface_hub 1.x | use `hf download`, or ModelScope |
| Download "stalled" at constant directory size | ModelScope preallocates `.incomplete` files | watch suffix removal, not `du` |
| `ImportError: is_torch_fx_available` loading the checkpoint | bundled remote code vs. current transformers | `trust_remote_code=False` (native class) |
| Both RoPE conventions pass the gate | golden prompt too short | regenerate with a ~150-word prompt |
| `fatal: detected dubious ownership` | rsync'd repo owned by different uid | `git config --global --add safe.directory '*'` |
| Everything slow at once | goldens generation (63 GB fp32 CPU load) saturates disk/memory bandwidth alongside the 31 GB serve load | serialize the heavy jobs; loader now logs per-layer progress so a stall names its layer |
| `pkill -f "<pattern>"` over ssh kills the session itself | the pattern matches the remote shell's own cmdline | run pkill from a script file, or use `pkill -f '[h]f download'` style patterns |

## 9. Session status snapshot (2026-08-09)

Done: environment up, 28/28 kernel tests green on the 4090, `inferx-serve`
built, both checkpoints down, goldens generated, RoPE convention measured —
**deinterleaved wins** pending the fp32 confirmation run. In flight when this
was written: fp32-reference gate, first serve smoke test, vLLM benchmark.
After confirmation: hardcode the deinterleaved convention, delete the env
toggle, commit the loader-logging change, and record serving/benchmark
numbers here.
