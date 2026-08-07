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
  libssl-dev
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
