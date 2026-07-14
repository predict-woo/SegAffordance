#!/usr/bin/env bash
# Bootstrap a RunPod pod for SegAffordance. Idempotent — safe to re-run.
# Usage: bash runpod/setup.sh  (from the repo root or anywhere)
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Persistent layout on the network volume
mkdir -p /workspace/datasets /workspace/checkpoints /workspace/runs /workspace/models /workspace/cache

# Basic tooling (base image is minimal)
if ! command -v tmux >/dev/null 2>&1 || ! command -v rsync >/dev/null 2>&1 || ! command -v htop >/dev/null 2>&1; then
  apt-get update -qq
  apt-get install -y -qq rsync tmux htop >/dev/null
fi

# torch/CUDA come from the base image; requirements.txt is the rest.
# The ubuntu2404 image enforces PEP 668 but ships torch in the system
# site-packages, so install alongside it.
pip install --no-cache-dir --break-system-packages -r "$REPO_DIR/requirements.txt"

# Keep model/dataset caches and wandb logs on the persistent volume
if ! grep -q "SegAffordance env" ~/.bashrc 2>/dev/null; then
  cat >> ~/.bashrc <<'EOF'

# SegAffordance env
export HF_HOME=/workspace/cache/huggingface
export TORCH_HOME=/workspace/cache/torch
export WANDB_DIR=/workspace/runs/wandb
EOF
fi

python - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("device:", torch.cuda.get_device_name() if torch.cuda.is_available() else "NO GPU")
PY

echo "setup.sh done. Open a new shell (or 'source ~/.bashrc') to pick up cache env vars."
