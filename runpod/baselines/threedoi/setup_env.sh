#!/usr/bin/env bash
# 3DOI (monoarti) env. Plain torch + SAM ViT-B, no custom CUDA ops, so it runs on the image's torch
# (dev pod: torch 2.9.1+cu128 Blackwell; training pod: whatever runpod/baselines/pods.sh picks).
# Installed into a pod-local venv that inherits the image's torch. Idempotent.
set -euo pipefail
B=/workspace/datasets/baselines; R=$B/repos; CK=$B/ckpt; mkdir -p $R $CK $B/logs $B/runs /workspace/tmp
export TMPDIR=/workspace/tmp
VENV=/opt/venv_3doi
[ -x $VENV/bin/python ] || python -m venv --system-site-packages $VENV
$VENV/bin/python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.device_count(), 'gpus')"
# their README list minus pytorch3d (3D viz only) and visdom/submitit (import-guarded by patch_train.py)
# torchvision matching the image's torch (the RunPod images ship torch without torchvision)
$VENV/bin/python -c "import torchvision" 2>/dev/null || $VENV/bin/pip install -q torchvision "torch==$($VENV/bin/python -c 'import torch; print(torch.__version__.split("+")[0])')" --index-url https://download.pytorch.org/whl/$($VENV/bin/python -c 'import torch; print("cu" + torch.version.cuda.replace(".", ""))')
PKGS="accelerate hydra-core omegaconf scipy submitit pycocotools packaging plotly imageio imageio-ffmpeg matplotlib h5py opencv-python-headless tqdm wandb lmdb"
# checked by import, not by a stamp file: the package list has grown more than once
$VENV/bin/python -c "import accelerate, hydra, scipy, submitit, pycocotools, cv2, h5py, wandb, lmdb" 2>/dev/null || $VENV/bin/pip install -q $PKGS
[ -d $R/3DOI/.git ] || git clone -q https://github.com/JasonQSY/3DOI $R/3DOI
cd $R/3DOI && git checkout -q 9ff32ef && echo "3DOI @ $(git rev-parse --short HEAD)"
python /workspace/SegAffordance/runpod/baselines/threedoi/patch_train.py $R/3DOI/monoarti
cp /workspace/SegAffordance/config/baselines/3doi_sam_sf3d.yaml $R/3DOI/monoarti/configs/sam_sf3d.yaml
# SAM ViT-B initial weights (their model.py loads checkpoints/sam_vit_b_01ec64.pth when sam_pretrained)
mkdir -p $R/3DOI/monoarti/checkpoints
[ -f $CK/sam_vit_b_01ec64.pth ] || curl -sL -o $CK/sam_vit_b_01ec64.pth https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
ln -sfn $CK/sam_vit_b_01ec64.pth $R/3DOI/monoarti/checkpoints/sam_vit_b_01ec64.pth
# dataset root: their hard-coded /home/ubuntu/monoarti_data -> our converted tree
mkdir -p /home/ubuntu && ln -sfn $B/stage/3doi /home/ubuntu/monoarti_data
cd $R/3DOI/monoarti && $VENV/bin/python -c "
import sys; sys.path.insert(0, '.')
from monoarti.model import build_model; from monoarti import dataset; print('3doi import ok', dataset.DEFAULT_DATA_ROOT)"
echo "== 3doi env done"
