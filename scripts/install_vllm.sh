#!/bin/bash
# Install vLLM into its own venv (separate from sglang to avoid torch-version
# conflicts). Logs to /tmp/vllm_install.log. Background-safe.
set -e
cd ~/inferx
echo "[*] creating .venv-vllm"
python3 -m venv .venv-vllm
source .venv-vllm/bin/activate
pip install --upgrade pip wheel
echo "[*] installing vllm (this pulls CUDA torch; several minutes)"
pip install vllm
echo "[*] verifying"
python -c 'import vllm, torch; print("vllm", vllm.__version__, "torch", torch.__version__, "cuda", torch.cuda.is_available())'
echo "[DONE] vllm ready"
