#!/usr/bin/env bash
# OPDMulti (OPDFormer) environment on runpod/pytorch:2.1.1-py3.10-cuda12.1.1-devel-ubuntu22.04.
# Upstream pins torch 1.10 / detectron2 cu111 wheels; we build detectron2 from
# source against the image's torch 2.1.1 (A100). Idempotent (markers in /root).
# The repo clone lives on the shared volume; the MSDeformAttn op is built in a
# per-pod /tmp copy so concurrent pods never write into the same build dir.
set -euo pipefail
B=/workspace/datasets/baselines; R=$B/repos; mkdir -p $R $B/data $B/runs $B/logs $B/ckpt /workspace/tmp
export CUDA_HOME=/usr/local/cuda PATH=/usr/local/cuda/bin:$PATH
export PIP_BREAK_SYSTEM_PACKAGES=1   # Ubuntu 24.04 images (PEP 668) refuse system pip otherwise
CC_CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d ' ')
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-${CC_CAP:-8.0}}" MAX_JOBS=${MAX_JOBS:-16}
echo "== GPU"; nvidia-smi --query-gpu=name,memory.total,clocks.max.sm,power.limit --format=csv
python -c "import torch;print('torch',torch.__version__,'cuda',torch.version.cuda,torch.cuda.is_available())"
echo "== torch/toolkit match"; TK=$(/usr/local/cuda/bin/nvcc --version | grep -o 'release [0-9.]*' | awk '{print $2}')
TV=$(python -c "import torch;print(torch.version.cuda)")
if [ "$TK" != "$TV" ] && [ ! -f /root/.done_torch_match ]; then
  # e.g. runpod/pytorch:1.0.3-cu1281-torch291 actually ships torch 2.12+cu130 on a 12.8 toolkit:
  # CUDA extensions (detectron2, MSDeformAttn) refuse to build. Install the toolkit-matching torch.
  echo "torch cuda $TV != toolkit $TK -> installing torch 2.8.0 for cu$(echo $TK | tr -d .)"
  pip install -q "torch==2.8.0" "torchvision==0.23.0" --index-url "https://download.pytorch.org/whl/cu$(echo $TK | tr -d .)" > /tmp/torch_install.log 2>&1 || { tail -5 /tmp/torch_install.log; exit 1; }
  touch /root/.done_torch_match
fi
python -c "import torch;print('torch',torch.__version__,'cuda',torch.version.cuda,torch.cuda.is_available())"
echo "== apt"; [ -f /root/.done_apt ] || { apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq libgl1 libglib2.0-0 ninja-build build-essential git >/dev/null && touch /root/.done_apt; }
echo "== pip"; if [ ! -f /root/.done_pip_opd ]; then
  pip install -q "numpy<2" opencv-python-headless h5py timm scipy shapely scikit-image cython pycocotools lmdb pyyaml "setuptools<70" wheel
  pip install -q git+https://github.com/cocodataset/panopticapi.git
  pip install -q --no-build-isolation 'git+https://github.com/facebookresearch/detectron2.git' > /tmp/d2_build.log 2>&1 || { tail -30 /tmp/d2_build.log; exit 1; }
  touch /root/.done_pip_opd; fi
python -c "import detectron2; print('detectron2', detectron2.__version__)"
echo "== repo"; REPO="${OPD_REPO:-OPDMulti}"; URL="${OPD_URL:-https://github.com/3dlg-hcvc/OPDMulti}"
[ -d $R/$REPO/.git ] || git clone -q "$URL" $R/$REPO
cd $R/$REPO && echo "$REPO @ $(git rev-parse --short HEAD)"
python /workspace/SegAffordance/runpod/baselines/opd/patch_mapper.py $R/$REPO/opdformer/mask2former/data/motion_dataset_mapper.py
python /workspace/SegAffordance/runpod/baselines/opd/patch_numpy_aliases.py $R/$REPO/opdformer/mask2former
echo "== MSDeformAttn"; if ! python -c "import torch, MultiScaleDeformableAttention" 2>/dev/null; then
  rm -rf /tmp/msda && cp -r $R/$REPO/opdformer/mask2former/modeling/pixel_decoder/ops /tmp/msda && cd /tmp/msda
  # torch >= 2.x removed the DeprecatedTypeProperties dispatch: AT_DISPATCH_*(value.type(), ...) -> value.scalar_type()
  sed -i 's/AT_DISPATCH_FLOATING_TYPES(value\.type()/AT_DISPATCH_FLOATING_TYPES(value.scalar_type()/g' src/cuda/ms_deform_attn_cuda.cu
  python setup.py build install > /tmp/msda_build.log 2>&1 || { tail -30 /tmp/msda_build.log; exit 1; }
fi
python -c "import torch, MultiScaleDeformableAttention; print('MSDeformAttn ok')"
echo "== env done ($REPO)"
