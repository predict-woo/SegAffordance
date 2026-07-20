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

# Configs load CLIP weights from pretrain/RN50.pt; the real file lives on the
# volume (not in git). Download it there once if missing, then symlink.
if [ ! -f /workspace/models/RN50.pt ]; then
  curl -fL -o /workspace/models/RN50.pt \
    "https://openaipublic.azureedge.net/clip/models/afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762/RN50.pt"
fi
mkdir -p "$REPO_DIR/pretrain"
ln -sf /workspace/models/RN50.pt "$REPO_DIR/pretrain/RN50.pt"

# Python deps live in a venv ON THE VOLUME: RunPod wipes the container disk
# on every pod stop, so system-site pip installs vanish across restarts.
# The venv inherits torch/CUDA from the base image via --system-site-packages
# and only needs to be built once per volume.
VENV=/workspace/venv
if [ ! -x "$VENV/bin/python" ]; then
  python3 -m venv --system-site-packages "$VENV"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"

pip install --no-cache-dir -r "$REPO_DIR/requirements.txt"

# torchvision is not in the base image; pin to the build matching its torch.
pip install --no-cache-dir \
  torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu128

# detectron2 has no wheel and its setup.py imports torch, so it must be
# built without pip's build isolation (after torch is present).
python -c 'import detectron2' 2>/dev/null || \
  pip install --no-cache-dir --no-build-isolation \
    'git+https://github.com/facebookresearch/detectron2.git'

# Keep model/dataset caches and wandb logs on the persistent volume, and
# activate the venv in interactive shells. (Non-interactive ssh commands
# should call /workspace/venv/bin/python explicitly.)
if ! grep -q "SegAffordance env" ~/.bashrc 2>/dev/null; then
  cat >> ~/.bashrc <<'EOF'

# SegAffordance env
export HF_HOME=/workspace/cache/huggingface
export TORCH_HOME=/workspace/cache/torch
export WANDB_DIR=/workspace/runs/wandb
export CODEX_HOME=/workspace/.codex
export PATH="/workspace/bin:$PATH"
source /workspace/venv/bin/activate
EOF
fi

python - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("device:", torch.cuda.get_device_name() if torch.cuda.is_available() else "NO GPU")
PY

echo "setup.sh done. Open a new shell (or 'source ~/.bashrc') to pick up cache env vars."
