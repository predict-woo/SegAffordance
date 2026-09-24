#!/bin/bash
# baseline -> teacher_forcing (dct_baseline already trained), test passes for
# all three, then SF3D 3D-DCT post-training initialized from the better of
# dct_baseline / teacher_forcing (held-out mIoU*100 + PDet), then its test.
cd /workspace/SegAffordance
T() { # arm exp
  exp=experiments/$1; ck=$(ls $exp/checkpoints/best-*.ckpt | head -1)
  echo "=== TEST $1 $ck $(date)"
  HF_HOME=/root/hfcache HF_HUB_OFFLINE=1 /opt/venv/bin/python train_SF3D_better.py test --config $exp/config.yaml --ckpt_path $ck --trainer.logger=false --data.lmdb_path /dev/shm/data.lmdb --data.frame_cache_path /dev/shm/frames.lmdb --data.num_workers_val 8 > $exp/logs/test.log 2>&1
  echo "=== TESTDONE $1 exit=$?"
}
bash runpod/sweep_queue.sh 20260907_hoi4d_2d_v2_baseline:config/hoi4d_v2_baseline.yaml 20260907_hoi4d_2d_v2_teacher_forcing:config/hoi4d_v2_teacher_forcing.yaml
for a in dct_baseline baseline teacher_forcing; do T 20260907_hoi4d_2d_v2_$a; done
echo HOI4D_ARMS_DONE
pick=$(/opt/venv/bin/python - <<'PY'
import re
def score(a):
    try: t=open(f"experiments/20260907_hoi4d_2d_v2_{a}/logs/test.log").read().replace("\r","\n")
    except FileNotFoundError: return None
    g=lambda k: float(re.search(rf"test/{k}\s+([0-9.]+)", t).group(1))
    return g("mean_iou")*100 + g("p_det")
s={a:score(a) for a in ("dct_baseline","teacher_forcing")}
print("teacher_forcing" if (s["teacher_forcing"] or -1) > (s["dct_baseline"] or -1) else "dct_baseline")
PY
)
exp=experiments/20260907_hoi4d_2d_v2_$pick; ck=$(ls $exp/checkpoints/best-*.ckpt | head -1)
echo "=== SF3D INIT from $pick: $ck"
sed -i "s#finetune_from_path: \".*\"#finetune_from_path: \"/workspace/SegAffordance/$ck\"#" config/sf3d_train_runpod_g19_dct_ft_hoi4d.yaml experiments/20260907_sf3d_g19_dct_ft_hoi4d/config.yaml
grep finetune_from_path config/sf3d_train_runpod_g19_dct_ft_hoi4d.yaml
bash runpod/sweep_queue.sh 20260907_sf3d_g19_dct_ft_hoi4d:config/sf3d_train_runpod_g19_dct_ft_hoi4d.yaml
exp=experiments/20260907_sf3d_g19_dct_ft_hoi4d; ck=$(ls $exp/checkpoints/best-*.ckpt | head -1)
echo "=== TEST sf3d_ft $ck $(date)"
HF_HOME=/root/hfcache HF_HUB_OFFLINE=1 /opt/venv/bin/python train_SF3D_better.py test --config $exp/config.yaml --ckpt_path $ck --trainer.logger=false --data.lmdb_path /dev/shm/data.lmdb --data.frame_cache_path /dev/shm/frames.lmdb > $exp/logs/test.log 2>&1
echo "=== TESTDONE sf3d_ft exit=$?"; echo SF3D_FT_DONE
