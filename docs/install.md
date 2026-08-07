# Installation

## Asynchronous HTTP dependencies

The Boost.Beast/Folly HTTP foundation is enabled by default. On Ubuntu 24.04,
install Folly's required platform development packages before configuring
InferX:

```bash
sudo apt-get update
sudo apt-get install -y \
  libfmt-dev \
  libgoogle-glog-dev \
  libgflags-dev \
  libevent-dev \
  libssl-dev \
  protobuf-compiler \
  libprotobuf-dev \
  protobuf-compiler-grpc \
  libgrpc++-dev
```

The protobuf packages provide the C++ compiler and runtime; the gRPC packages
provide `grpc_cpp_plugin`, headers, and libraries for the optional
process-separated scheduler client. Verify both code generators before
configuring:

```bash
protoc --version
command -v grpc_cpp_plugin
```

On Ubuntu 24.04 systems where APT packages cannot be installed system-wide,
download and extract the packages into the ignored `.tools/grpc` prefix:

```bash
mkdir -p .tools/grpc
package_dir=$(mktemp -d)
(
  cd "$package_dir"
  apt-get download \
    protobuf-compiler libprotobuf-dev protobuf-compiler-grpc libgrpc++-dev \
    libprotobuf32t64 libprotobuf-lite32t64 libprotoc32t64 \
    libgrpc29t64 libgrpc++1.51t64 libgrpc-dev \
    libabsl-dev libabsl20220623t64 \
    libc-ares-dev libcares2 libre2-dev libre2-10 \
    pkgconf pkgconf-bin libpkgconf3
)
for package_file in "$package_dir"/*.deb; do
  dpkg-deb -x "$package_file" "$PWD/.tools/grpc"
done
```

Configure the shell and CMake to use that local prefix:

```bash
export PATH="$PWD/.tools/grpc/usr/bin:$PATH"
export CMAKE_PREFIX_PATH="$PWD/.tools/grpc/usr${CMAKE_PREFIX_PATH:+:$CMAKE_PREFIX_PATH}"
export PKG_CONFIG_PATH="$PWD/.tools/grpc/usr/lib/x86_64-linux-gnu/pkgconfig${PKG_CONFIG_PATH:+:$PKG_CONFIG_PATH}"
export LIBRARY_PATH="$PWD/.tools/grpc/usr/lib/x86_64-linux-gnu${LIBRARY_PATH:+:$LIBRARY_PATH}"
export LD_LIBRARY_PATH="$PWD/.tools/grpc/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

The `.tools` directory is ignored by Git and is local to one checkout.
When invoking `protoc` manually, pass the local plugin explicitly if the
compiler does not resolve it through `PATH`:

```bash
protoc -I proto \
  --cpp_out=generated \
  --grpc_out=generated \
  --plugin=protoc-gen-grpc="$PWD/.tools/grpc/usr/bin/grpc_cpp_plugin" \
  proto/inference/scheduler/v1/scheduler.proto
```

InferX pins Boost and Folly as Git submodules. The repository uses GitHub SSH
URLs, so make sure `ssh -T git@github.com` succeeds, then initialize the pinned
sources and Boost's nested libraries:

```bash
git submodule sync --recursive
git submodule update --init --checkout \
  third_party/boost third_party/folly third_party/fast_float
git -C third_party/boost submodule update \
  --init --recursive --checkout --depth 1
```

Configure using a fresh build directory:

```bash
cmake -S . -B build
cmake --build build -j"$(nproc)"
```

## Host-only gateway

When protobuf and gRPC are available, a CUDA-free configuration builds the
process-separated HTTP gateway without model weights, an engine, a scheduler
queue, or KV-cache state:

```bash
cmake -S . -B build-gateway \
  -DINFERX_ENABLE_CUDA=OFF \
  -DCMAKE_PREFIX_PATH="$CMAKE_PREFIX_PATH"
cmake --build build-gateway --target inferx-gateway -j"$(nproc)"
```

The gateway loads only the CPU tokenizer artifacts from the checkpoint and
sends tokenized requests to the scheduler endpoint:

```bash
./build-gateway/src/server/gateway/inferx-gateway \
  --scheduler-endpoint dns:///scheduler:50051 \
  --tokenizer /models/example \
  --model example \
  --model-version v1 \
  --tokenizer-revision tokenizer-v1
```

The tokenizer dependency requires a working Rust `cargo` executable when its
vendored static library has not already been built.
