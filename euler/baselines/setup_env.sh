#!/usr/bin/env bash
# Build both baseline environments on Euler in plain venvs. pip's CUDA wheels bundle their own
# CUDA runtime, so only the NVIDIA DRIVER version on the compute nodes matters -- no cuda module
# is loaded here (Euler currently offers only cuda/13.0.2, which is newer than either stack).
#   bash setup_env.sh [a3vlm|3doi|both]
set -euo pipefail
WHAT="${1:-both}"
BASE="${EULER_BASE:-$SCRATCH/baselines}"
R=$BASE/repos; mkdir -p $R $BASE/logs $BASE/runs
export TMPDIR="${TMPDIR:-$SCRATCH/tmp}"; mkdir -p "$TMPDIR"
export PIP_CACHE_DIR="$BASE/.pipcache"
module load stack/2024-06 python/3.11.6 2>/dev/null || module load python 2>/dev/null || true
echo "python: $(python3 -V) at $(command -v python3)"

if [ "$WHAT" = "a3vlm" ] || [ "$WHAT" = "both" ]; then
  # A3VLM pins torch 2.0.1 (cu118). Its cp310 wheels need python 3.10; on a 3.11+ stack the
  # nearest workable build is torch 2.1.2+cu118, which runs llama_ens5 unchanged.
  V=$BASE/venv_a3vlm
  [ -x $V/bin/python ] || python3 -m venv $V
  $V/bin/pip install -q --upgrade pip
  $V/bin/python -c "import torch" 2>/dev/null || \
    $V/bin/pip install -q torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118
  $V/bin/python -c "import fairscale, sentencepiece, transformers, open_clip, timm, dacite" 2>/dev/null || \
    $V/bin/pip install -q fairscale sentencepiece "transformers==4.34.1" "open_clip_torch==2.23.0" "timm==0.9.12" \
      fire packaging pyyaml Ninja einops regex h5py pandas pyarrow tensorboard "httpx[socks]" tqdm dacite \
      pycocotools opencv-python-headless lmdb
  # the fork is a partial tree: overlay it on upstream LLaMA2-Accessory (see runpod/baselines/a3vlm/setup_env.sh)
  [ -d $R/A3VLM/.git ] || git clone -q https://github.com/changhaonan/A3VLM $R/A3VLM
  (cd $R/A3VLM && git checkout -q 436e715)
  [ -d $R/LLaMA2-Accessory/.git ] || git clone -q https://github.com/Alpha-VLLM/LLaMA2-Accessory $R/LLaMA2-Accessory
  (cd $R/LLaMA2-Accessory && git checkout -q $(git -C $R/LLaMA2-Accessory rev-list -1 --before=2024-10-08 main))
  cp -r $R/A3VLM/model/accessory/. $R/LLaMA2-Accessory/accessory/
  [ -f $R/LLaMA2-Accessory/accessory/configs/global_configs.py ] || \
    printf 'try:\n    import flash_attn\n    USE_FLASH_ATTENTION = True\nexcept ImportError:\n    USE_FLASH_ATTENTION = False\n' \
      > $R/LLaMA2-Accessory/accessory/configs/global_configs.py
  export TORCH_HOME=$BASE/ckpt/torch; mkdir -p $TORCH_HOME/hub
  [ -d $TORCH_HOME/hub/facebookresearch_dinov2_main ] || git clone -q https://github.com/facebookresearch/dinov2 $TORCH_HOME/hub/facebookresearch_dinov2_main
  (cd $R/LLaMA2-Accessory/accessory && PYTHONPATH=$R/LLaMA2-Accessory $V/bin/python -c \
    "from accessory.model.LLM import llama_ens5; from accessory.model.meta import MetaModel; print('a3vlm import ok')")
fi

if [ "$WHAT" = "3doi" ] || [ "$WHAT" = "both" ]; then
  V=$BASE/venv_3doi
  [ -x $V/bin/python ] || python3 -m venv $V
  $V/bin/pip install -q --upgrade pip
  $V/bin/python -c "import torch, torchvision" 2>/dev/null || $V/bin/pip install -q torch torchvision
  $V/bin/python -c "import accelerate, hydra, scipy, submitit, pycocotools, cv2, h5py, wandb, lmdb" 2>/dev/null || \
    $V/bin/pip install -q accelerate hydra-core omegaconf scipy submitit pycocotools packaging plotly imageio \
      imageio-ffmpeg matplotlib h5py opencv-python-headless tqdm wandb lmdb
  [ -d $R/3DOI/.git ] || git clone -q https://github.com/JasonQSY/3DOI $R/3DOI
  (cd $R/3DOI && git checkout -q 9ff32ef)
  python3 "$(dirname "$0")/../../runpod/baselines/threedoi/patch_train.py" $R/3DOI/monoarti
  cp "$(dirname "$0")/../../config/baselines/3doi_sam_sf3d.yaml" $R/3DOI/monoarti/configs/sam_sf3d.yaml
  mkdir -p $R/3DOI/monoarti/checkpoints
  [ -f $BASE/ckpt/sam_vit_b_01ec64.pth ] || curl -sL -o $BASE/ckpt/sam_vit_b_01ec64.pth https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
  ln -sfn $BASE/ckpt/sam_vit_b_01ec64.pth $R/3DOI/monoarti/checkpoints/sam_vit_b_01ec64.pth
  # their dataset root is hard-coded; on Euler we cannot write /home/ubuntu, so symlink inside $HOME
  mkdir -p $HOME/.monoarti && ln -sfn $BASE/data/3doi $HOME/.monoarti/monoarti_data
  (cd $R/3DOI/monoarti && $V/bin/python -c "
import re, pathlib
p = pathlib.Path('monoarti/dataset.py'); s = p.read_text()
import os
root = os.path.expanduser('~/.monoarti/monoarti_data')
if root not in s:
    s = s.replace(\"DEFAULT_DATA_ROOT = '/home/ubuntu/monoarti_data'\", f\"DEFAULT_DATA_ROOT = '{root}'\")
    s = s.replace(\"DEFAULT_DEPTH_ROOT = '/home/ubuntu/monoarti_data/omnidata_filtered/depth_zbuffer/taskonomy'\", f\"DEFAULT_DEPTH_ROOT = '{root}/omnidata_filtered/depth_zbuffer/taskonomy'\")
    p.write_text(s); print('dataset root ->', root)
")
  (cd $R/3DOI/monoarti && $V/bin/python -c "import sys; sys.path.insert(0,'.'); from monoarti.model import build_model; from monoarti import dataset; print('3doi import ok', dataset.DEFAULT_DATA_ROOT)")
fi
echo "== euler env(s) ready under $BASE"
