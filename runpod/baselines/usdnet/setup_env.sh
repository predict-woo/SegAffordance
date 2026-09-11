#!/usr/bin/env bash
# USDNet env on runpod/pytorch:2.1.1-py3.10-cuda12.1.1-devel-ubuntu22.04 (torch 2.1.1+cu121).
# Copy of runpod/external/usdnet/setup_env.sh (2026-08-29, proven on A100) with
# the baseline paths and without the screw-loss patch. Markers in /root.
set -euo pipefail
B=/workspace/datasets/baselines; W=$B/usdnet; R=$B/repos/USDNet; mkdir -p $W $B/repos $B/data $B/runs $B/logs $B/ckpt /workspace/tmp
export CUDA_HOME=/usr/local/cuda PATH=/usr/local/cuda/bin:$PATH LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0;9.0}" MAX_JOBS=${MAX_JOBS:-16} OMP_NUM_THREADS=3
echo "== GPU"; nvidia-smi --query-gpu=name,memory.total,clocks.max.sm,power.limit --format=csv; nvcc --version | tail -1
python -c "import torch;print('torch',torch.__version__,'cuda',torch.version.cuda,torch.cuda.is_available())"
echo "== repo"; [ -d $R/.git ] || git clone -q --recursive https://github.com/insait-institute/USDNet $R
cd $R && git checkout -q 0ba303d90375b86f61fb79007f6ff9757f5f2c15 && echo "USDNet @ $(git rev-parse --short HEAD)"
echo "== apt"; [ -f /root/.done_apt ] || { apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq libopenblas-dev libblas-dev liblapack-dev libgl1 libglib2.0-0 build-essential ninja-build git >/dev/null && touch /root/.done_apt; }
echo "== pip"; if [ ! -f /root/.done_pip_usd ]; then
  python -m pip install -q "pip==23.1" "setuptools<70" wheel && pip install -q --ignore-installed blinker
  python - <<'PY' > /tmp/req.txt
import re
lines=open('environment.yml').read().split('- pip:')[1].splitlines()
skip=('torch','detectron2','jupyter','ipykernel','ipython','ipywidgets','black','flake8','dash','debugpy','nbformat','nbconvert','notebook','widgetsnbextension','mypy','pytorch-lightning','torchmetrics','numpy','MinkowskiEngine','pyviz3d','open3d')
for l in lines:
    l=l.strip().lstrip('- ').strip()
    if not l or l.startswith('#') or ':' in l: continue
    if any(l.lower().startswith(s.lower()) for s in skip): continue
    print(l)
PY
  pip install -q -r /tmp/req.txt || { echo "bulk install failed; retrying per package"; while read -r p; do pip install -q "$p" || echo "SKIP $p"; done < /tmp/req.txt; }
  pip install -q "numpy<2" cython pycocotools omegaconf==2.0.6 hydra-core==1.0.5 pytorch-lightning==1.7.2 torchmetrics==0.9.3 h5py python-dotenv usd-core open3d pyviz3d lmdb scipy
  pip install -q torch-scatter -f https://data.pyg.org/whl/torch-2.1.0+cu121.html
  pip install -q 'git+https://github.com/facebookresearch/detectron2.git@710e7795d0eeadf9def0e7ef957eea13532e34cf' --no-deps
  touch /root/.done_pip_usd; fi
echo "== MinkowskiEngine"; if ! python -c "import MinkowskiEngine" 2>/dev/null; then
  mkdir -p /tmp/me && cd /tmp/me && rm -rf MinkowskiEngine
  git clone -q --recursive https://github.com/NVIDIA/MinkowskiEngine && cd MinkowskiEngine && git checkout -q 02fc608bea4c0549b0a7b00ca1bf15dee4a0b228
  sed -i '1i #include <thrust/execution_policy.h>' src/convolution_kernel.cuh
  sed -i '1i #include <thrust/unique.h>\n#include <thrust/remove.h>' src/coordinate_map_gpu.cu
  sed -i '1i #include <thrust/execution_policy.h>\n#include <thrust/reduce.h>\n#include <thrust/sort.h>' src/spmm.cu
  sed -i '1i #include <thrust/execution_policy.h>' src/3rdparty/concurrent_unordered_map.cuh
  find src -type f \( -name "*.cu" -o -name "*.cuh" -o -name "*.hpp" \) | xargs sed -i 's/thrust::device/thrust::cuda::par/g'
  python setup.py install --force_cuda --blas=openblas > /tmp/me_build.log 2>&1 || { tail -30 /tmp/me_build.log; exit 1; }
  cd $R; fi
python -c "import MinkowskiEngine as ME; print('ME', ME.__version__)"
echo "== segmentator + pointnet2"; cd $R/third_party
[ -d ScanNet ] || git clone -q https://github.com/ScanNet/ScanNet.git
cd ScanNet/Segmentator && git checkout -q 3e5726500896748521a6ceb81271b0f5b2c0e7d2 && make -s && cd ../..
cd pointnet2 && python setup.py install -q > /tmp/pn2_build.log 2>&1 || tail -20 /tmp/pn2_build.log; cd $R
pip install -q pytorch-lightning==1.7.2
echo "== backbone ckpt"; cd $B/ckpt; [ -f scannet200_benchmark.ckpt ] || curl -sL -o scannet200_benchmark.ckpt https://omnomnom.vision.rwth-aachen.de/data/mask3d/checkpoints/scannet200/scannet200_benchmark.ckpt; ls -la scannet200_benchmark.ckpt
echo "== env done"
