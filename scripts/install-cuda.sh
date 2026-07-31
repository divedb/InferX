#!/usr/bin/env bash
#
# Installs the CUDA 13.x toolkit from NVIDIA's apt repository.
#
# Needs sudo. Run it yourself rather than from a build script -- it modifies
# apt sources and installs several GB.
#
# WSL2 note: install the *WSL-Ubuntu* toolkit variant, which deliberately omits
# the display driver. The GPU driver lives on the Windows host and is projected
# into WSL via /usr/lib/wsl/lib. Installing a Linux driver here will shadow that
# and break CUDA entirely.

set -euo pipefail

CUDA_PKG="${CUDA_PKG:-cuda-toolkit-13-0}"

log() { printf '\033[1;34m==>\033[0m %s\n' "$*"; }
err() { printf '\033[1;31merror:\033[0m %s\n' "$*" >&2; }

if [[ "$(uname -s)" != "Linux" ]]; then
  err "this script targets Linux (including WSL2)"
  exit 1
fi

# ---------------------------------------------------------------------------
# Driver check. The toolkit is useless without a new enough driver, and on WSL2
# that driver can only be updated from Windows -- so fail early with the right
# instruction rather than after a 4 GB download.
# ---------------------------------------------------------------------------
if command -v nvidia-smi >/dev/null 2>&1; then
  driver_cuda="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 || true)"
  log "detected NVIDIA driver: ${driver_cuda:-unknown}"
  log "driver-reported max CUDA: $(nvidia-smi 2>/dev/null | grep -oP 'CUDA Version: \K[0-9.]+' | head -1 || echo unknown)"
else
  err "nvidia-smi not found."
  if grep -qi microsoft /proc/version 2>/dev/null; then
    err "On WSL2, install/update the NVIDIA driver on the WINDOWS host, not here."
  fi
  exit 1
fi

is_wsl=0
grep -qi microsoft /proc/version 2>/dev/null && is_wsl=1

# ---------------------------------------------------------------------------
# Repository selection.
# ---------------------------------------------------------------------------
. /etc/os-release
distro_id="${ID}${VERSION_ID//./}"   # e.g. ubuntu2404
arch="$(dpkg --print-architecture)"
case "$arch" in
  amd64) repo_arch="x86_64" ;;
  arm64) repo_arch="sbsa" ;;
  *) err "unsupported architecture: $arch"; exit 1 ;;
esac

if [[ $is_wsl -eq 1 ]]; then
  repo_distro="wsl-ubuntu"
  log "WSL2 detected -- using the wsl-ubuntu repo (no display driver)"
else
  repo_distro="$distro_id"
fi

base_url="https://developer.download.nvidia.com/compute/cuda/repos/${repo_distro}/${repo_arch}"

log "installing keyring from ${base_url}"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
curl -fsSL -o "$tmp/cuda-keyring.deb" "${base_url}/cuda-keyring_1.1-1_all.deb"
sudo dpkg -i "$tmp/cuda-keyring.deb"

log "updating package lists"
sudo apt-get update

log "installing ${CUDA_PKG}"
sudo apt-get install -y "$CUDA_PKG"

# ---------------------------------------------------------------------------
# The distro's own nvidia-cuda-toolkit package installs /usr/bin/nvcc, which
# takes precedence over /usr/local/cuda/bin on the default PATH. Leaving both
# installed is the single most common cause of "I installed 13 but CMake still
# finds 12".
# ---------------------------------------------------------------------------
if dpkg -l nvidia-cuda-toolkit 2>/dev/null | grep -q '^ii'; then
  err "the distro package 'nvidia-cuda-toolkit' is installed and its"
  err "/usr/bin/nvcc will shadow the new toolkit."
  err "Remove it with:  sudo apt-get remove -y nvidia-cuda-toolkit"
fi

cuda_root="$(ls -d /usr/local/cuda-13* 2>/dev/null | sort -V | tail -1 || true)"
if [[ -z "$cuda_root" ]]; then
  err "no /usr/local/cuda-13* found after install"
  exit 1
fi

log "installed: $("$cuda_root/bin/nvcc" --version | tail -2 | head -1)"

cat <<EOF

Add to your shell profile:

  export CUDA_HOME=$cuda_root
  export PATH=\$CUDA_HOME/bin:\$PATH
  export LD_LIBRARY_PATH=\$CUDA_HOME/lib64:\${LD_LIBRARY_PATH:-}

Then verify:

  nvcc --version          # must report 13.x
  ./scripts/bootstrap.sh

EOF
