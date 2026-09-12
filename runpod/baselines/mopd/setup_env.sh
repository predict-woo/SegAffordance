#!/usr/bin/env bash
# MOPD (Locate n' Rotate) env = OPDMulti env + geffnet/pot/uotod, its own repo
# clone and its own MSDeformAttn build. Run AFTER runpod/baselines/opd/setup_env.sh.
set -euo pipefail
B=/workspace/datasets/baselines; R=$B/repos; mkdir -p $R $B/ckpt
export CUDA_HOME=/usr/local/cuda PATH=/usr/local/cuda/bin:$PATH PIP_BREAK_SYSTEM_PACKAGES=1
CC_CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d ' ')
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-${CC_CAP:-8.0}}" MAX_JOBS=${MAX_JOBS:-16}
[ -f /root/.done_pip_opd ] || { echo "run opd/setup_env.sh first"; exit 1; }
[ -f /root/.done_pip_mopd ] || { pip install -q geffnet pot matplotlib && (pip install -q uotod || echo "uotod install failed (matcher patch will be applied)") && touch /root/.done_pip_mopd; }
[ -d $R/MOPD/.git ] || git clone -q https://github.com/lisiqi-zju/MOPD $R/MOPD
cd $R/MOPD && echo "MOPD @ $(git rev-parse --short HEAD)"
python /workspace/SegAffordance/runpod/baselines/opd/patch_mapper.py $R/MOPD/opdformer/mask2former/data/motion_dataset_mapper.py
python /workspace/SegAffordance/runpod/baselines/opd/patch_numpy_aliases.py $R/MOPD/opdformer/mask2former
# MSDeformAttn: the same op as OPDMulti's (byte-identical ops dir); build once per pod.
python -c "import torch, MultiScaleDeformableAttention" 2>/dev/null || { rm -rf /tmp/msda && cp -r $R/MOPD/opdformer/mask2former/modeling/pixel_decoder/ops /tmp/msda && cd /tmp/msda && python setup.py build install > /tmp/msda_build.log 2>&1; }
python -c "import torch, MultiScaleDeformableAttention, geffnet; print('mopd deps ok')"
python -c "import uotod" 2>/dev/null || echo "NOTE: uotod unusable -> matcher patched to skip its (discarded) Sinkhorn block"
python /workspace/SegAffordance/runpod/baselines/mopd/patch_matcher.py $R/MOPD/opdformer/mask2former/modeling/matcher.py
python /workspace/SegAffordance/runpod/baselines/mopd/patch_model_load.py $R/MOPD/opdformer/mask2former/maskformer_model.py
# EfficientSAM ViT-S release weights (zip on GitHub; no `unzip` binary on the images -> python)
cd $B/ckpt; [ -f efficient_sam_vits.pt ] || { curl -sL -o efficient_sam_vits.pt.zip https://github.com/yformer/EfficientSAM/raw/main/weights/efficient_sam_vits.pt.zip && python -c "import zipfile; zipfile.ZipFile('efficient_sam_vits.pt.zip').extractall('.')" && rm -f efficient_sam_vits.pt.zip; }
ls -la $B/ckpt/efficient_sam_vits.pt
echo "== mopd env done"
