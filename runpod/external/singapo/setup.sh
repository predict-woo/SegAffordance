#!/usr/bin/env bash
# On-pod setup for the SINGAPO A/B. Idempotent: volume-level markers under
# /workspace for data/repo/ckpts, container-level markers under /root for env.
set -euo pipefail
W=/workspace; R=$W/repo/singapo; OURS=$W/ours
mkdir -p $W/repo $W/data $W/ckpt_init $W/results $W/logs
echo "== GPU"; nvidia-smi --query-gpu=name,memory.total,clocks.max.sm,power.limit --format=csv 2>/dev/null || echo "no GPU on this pod"
free -g | head -2; df -h $W | tail -1; ulimit -n 65536 || true

echo "== apt"; [ -f /root/.done_apt ] || { apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq graphviz libgraphviz-dev libegl1 libgl1 libglib2.0-0 unzip aria2 >/dev/null && touch /root/.done_apt; }

echo "== repo"; [ -d $R/.git ] || git clone -q https://github.com/3dlg-hcvc/singapo $R
cd $R && git checkout -q de4b46616a7ecb1290bb84e215da0facb765bda5 && git log -1 --format="%h %cd"

echo "== pip"; [ -f /root/.done_pip ] || { pip install -q --break-system-packages --ignore-installed blinker && pip install -q --break-system-packages torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu128 && pip install -q --break-system-packages "lightning==2.3.3" "numpy==1.26.4" omegaconf diffusers imageio plotly pyrender open3d trimesh numpy-quaternion pygraphviz openai networkx scipy tqdm pytest && touch /root/.done_pip; }
python -c "import torch,lightning,diffusers,pyrender,open3d,pygraphviz;print('torch',torch.__version__,'cuda',torch.cuda.is_available())"

echo "== data"; cd $W/data
if [ ! -f .done_pm ]; then
  aria2c -q -x8 -s8 -c https://aspis.cmpt.sfu.ca/projects/singapo/data/pm.zip
  unzip -l pm.zip | sed -n 4,8p; unzip -q -o pm.zip && rm -f pm.zip
  [ -d pm ] && [ ! -d Table ] && mv pm/* . && rmdir pm || true
  ls; touch .done_pm
fi
if [ ! -f .done_aug ]; then
  aria2c -q -x8 -s8 -c https://aspis.cmpt.sfu.ca/projects/singapo/data/augmented_train.zip
  unzip -l augmented_train.zip | sed -n 4,8p; unzip -q -o augmented_train.zip && rm -f augmented_train.zip
  for d in augmented_train aug; do [ -d $d ] && [ ! -d augmented_data ] && mv $d/* . && rmdir $d || true; done
  ls; touch .done_aug
fi
python - <<'PY'
import json,os
ids=json.load(open('/workspace/repo/singapo/data/data_split.json'))
miss=[i for s in ('train','test') for i in ids[s] if not os.path.exists(f'/workspace/data/{i}/object.json')]
print('split ids missing on disk:',len(miss), miss[:5])
PY

echo "== ckpts"; cd $W/ckpt_init
if [ ! -f .done ]; then
  aria2c -q -c https://aspis.cmpt.sfu.ca/projects/singapo/ckpts/singapo_ckpt.zip && unzip -q -o singapo_ckpt.zip
  aria2c -q -c https://aspis.cmpt.sfu.ca/projects/singapo/ckpts/pretrained_cage.zip && unzip -q -o pretrained_cage.zip
  touch .done
fi
find $W/ckpt_init -name "*.ckpt" | tee $W/ckpt_init/ckpt_list.txt
mkdir -p $R/pretrained && cp -n $(find $W/ckpt_init -name "*cage*.ckpt" | head -1) $R/pretrained/cage_cfg.ckpt 2>/dev/null || true

echo "== our files"; cp $OURS/screw_loss.py $OURS/test_screw_loss.py $OURS/train_ft.py $OURS/test_ft.py $OURS/eval_axis.py $R/
cp $OURS/system_screw.py $R/systems/; cp $OURS/finetune.yaml $R/configs/
cd $R && python -m pytest -q test_screw_loss.py 2>&1 | tail -1
echo "== setup done"
# pytorch3d (their CD metric) — no wheel for torch 2.9/cu128; source build (~20 min, needs nvcc)
python -c "import pytorch3d" 2>/dev/null || { export TORCH_CUDA_ARCH_LIST="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)" MAX_JOBS=32 FORCE_CUDA=1; pip install --break-system-packages --no-build-isolation "git+https://github.com/facebookresearch/pytorch3d.git@stable" > /workspace/logs/p3d_build.log 2>&1 && python -c "import pytorch3d;print('pytorch3d', pytorch3d.__version__)"; }
