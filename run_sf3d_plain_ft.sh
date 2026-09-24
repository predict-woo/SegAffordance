#!/bin/bash
# Pod-side: SF3D g19_dct post-training from the two PLAIN-head HOI4D v2 arms,
# back-to-back on one pod, then the test pass for each. Args: arm tags to run
# (default: both). Markers: SF3D_PLAIN_ARM_DONE <exp>, SF3D_PLAIN_ALL_DONE.
cd /workspace/SegAffordance
ARMS="${*:-ft_hoi4d_plain ft_hoi4d_tf_plain}"
# Resolve the baseline init checkpoint (name unknown Mac-side: volume-only).
bk=$(ls experiments/20260907_hoi4d_2d_v2_baseline/checkpoints/best-*.ckpt | head -1)
echo "=== baseline init ckpt: $bk"
for f in config/sf3d_train_runpod_g19_dct_ft_hoi4d_plain.yaml experiments/20260907_sf3d_g19_dct_ft_hoi4d_plain/config.yaml; do
  /opt/venv/bin/python - "$f" "/workspace/SegAffordance/$bk" <<'PY'
import sys; p, ck = sys.argv[1], sys.argv[2]
s = open(p).read().replace('"BASELINE_BEST_CKPT"', f'"{ck}"'); open(p, 'w').write(s)
PY
done
grep -n finetune_from_path config/sf3d_train_runpod_g19_dct_ft_hoi4d_plain.yaml config/sf3d_train_runpod_g19_dct_ft_hoi4d_tf_plain.yaml
T() { exp=experiments/$1; ck=$(ls $exp/checkpoints/best-*.ckpt | head -1); echo "=== TEST $1 $ck $(date)"
  HF_HOME=/root/hfcache HF_HUB_OFFLINE=1 /opt/venv/bin/python train_SF3D_better.py test --config $exp/config.yaml --ckpt_path $ck --trainer.logger=false --data.lmdb_path /dev/shm/data.lmdb --data.frame_cache_path /dev/shm/frames.lmdb > $exp/logs/test.log 2>&1; echo "=== TESTDONE $1 exit=$?"; }
for tag in $ARMS; do
  exp=20260907_sf3d_g19_dct_$tag
  bash runpod/sweep_queue.sh $exp:config/sf3d_train_runpod_g19_dct_$tag.yaml
  T $exp
  echo "SF3D_PLAIN_ARM_DONE $exp $(date)"
done
echo "SF3D_PLAIN_ALL_DONE $(date)"
