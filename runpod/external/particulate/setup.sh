#!/usr/bin/env bash
# Particulate setup. Phase A (CPU or GPU pod): repo, PartNet-Mobility download from HF
# (needs HF_TOKEN with access to sapien-sim/PartNetMobility), process_urdf + cache_points.
# Phase B (GPU pod): env + PartField ckpt + released model + patch + tests.
set -euo pipefail
W=/workspace; R=$W/repo/particulate; OURS=$W/ours; PHASE=${1:-all}
mkdir -p $W/repo $W/data $W/ckpt_init $W/results $W/logs
echo "== repo"; [ -d $R/.git ] || git clone -q --recursive https://github.com/RuiningLi/particulate $R
cd $R && git checkout -q dee37a75c449f324d9989993461ee09eaccc1686 && git log -1 --format="%h %cd"
pip install -q --break-system-packages huggingface_hub trimesh numpy scipy tqdm 2>/dev/null || pip install -q huggingface_hub trimesh numpy scipy tqdm
if [ "$PHASE" = all ] || [ "$PHASE" = data ]; then
  echo "== PartNet-Mobility (HF sapien-sim/PartNetMobility)"; cd $W/data
  if [ ! -f .done_pm_raw ]; then
    [ -n "${HF_TOKEN:-}" ] || { echo "HF_TOKEN not set"; exit 1; }
    python - <<'PY'
import os
from huggingface_hub import snapshot_download
p = snapshot_download("sapien-sim/PartNetMobility", repo_type="dataset", local_dir="/workspace/data/pm_zips", token=os.environ["HF_TOKEN"], max_workers=16)
print("downloaded to", p)
PY
    mkdir -p raw && cd pm_zips && ls *.zip | wc -l && for z in *.zip; do d=$W/data/raw/${z%.zip}; [ -f $d/mobility.urdf ] || { mkdir -p $d && unzip -q -o $z -d $d; }; done
    # some zips nest an <id>/ dir; flatten
    cd $W/data/raw && for d in */; do id=${d%/}; [ -f $id/mobility.urdf ] || { [ -f $id/$id/mobility.urdf ] && mv $id/$id/* $id/ && rmdir $id/$id; }; done
    ls | wc -l; touch $W/data/.done_pm_raw
  fi
  echo "== process_urdf + cache_points (train format, 40k pts)"; cd $R
  if [ ! -f $W/data/.done_cache ]; then
    mkdir -p $W/data/preprocessed $W/data/cached
    find $W/data/raw -mindepth 2 -maxdepth 2 -name mobility.urdf -print0 | xargs -0 -P "$(nproc)" -I{} bash -c 'u="{}"; o=$(basename $(dirname "$u")); [ -f /workspace/data/preprocessed/$o/link_axes_plucker.npz ] || python -m particulate.data.process_urdf "$u" /workspace/data/preprocessed/$o > /dev/null 2>&1 || echo "FAIL process $o"'
    find $W/data/preprocessed -mindepth 1 -maxdepth 1 -type d -print0 | xargs -0 -P "$(nproc)" -I{} bash -c 'd="{}"; b=$(basename "$d"); [ -f /workspace/data/cached/$b.npz ] || python -m particulate.data.cache_points --root "$d" --output_path /workspace/data/cached/$b.npz --num_points 40000 --ratio_sharp 0.5 > /dev/null 2>&1 || echo "FAIL cache $b"'
    ls $W/data/cached | wc -l; touch $W/data/.done_cache
  fi
fi
if [ "$PHASE" = all ] || [ "$PHASE" = env ]; then
  echo "== env"; cd $R
  [ -f /root/.done_pip ] || { pip install -q --break-system-packages --ignore-installed blinker; pip install -q --break-system-packages -r requirements.txt accelerate huggingface_hub pytest pandas 2>&1 | tail -2; touch /root/.done_pip; }
  python - <<'PY'
from huggingface_hub import hf_hub_download
import shutil, os
p = hf_hub_download(repo_id="rayli/Particulate", filename="model.pt"); shutil.copy(p, "/workspace/ckpt_init/model.pt")
os.makedirs("PartField/model", exist_ok=True)
hf_hub_download(repo_id="mikaelaangel/partfield-ckpt", filename="model_objaverse.ckpt", local_dir="PartField/model")
print("ckpts ok")
PY
  cp $OURS/screw_loss.py $OURS/test_screw_loss.py $OURS/apply_patch.py $R/ && python apply_patch.py && python -m pytest -q test_screw_loss.py 2>&1 | tail -1
  echo "== env done"
fi
