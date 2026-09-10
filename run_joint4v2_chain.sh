#!/bin/bash
# Pod J (2026-09-10 night): joint 2D+3D training v2 (DCT readout conventions v2). Stage SF3D (26 GB) + the three hand
# LMDBs into /dev/shm, train with a pod-local config copy, keep the best ckpt,
# test SF3D (pred_z_p + gt_z0) and each hand source with the single-source
# configs. Markers: JOINT_TRAIN_DONE, JOINT_TEST_DONE, CHAIN_DONE.
cd /workspace/SegAffordance
bash runpod/ensure_env.sh
# Lightning saves checkpoints via an fsspec transaction = a tempfile in $TMPDIR then a move;
# the pod overlay's /tmp can be smaller than a 3.7 GB ckpt (ENOSPC, dev pod 2026-09-10). Keep it on the volume.
export TMPDIR=/workspace/tmp; mkdir -p $TMPDIR
cat /workspace/cache/dinov3/*.pth > /dev/null 2>&1 || true
ulimit -n 65536
E=20260910_joint4_dctv2_rgb_scalefree; CFG=config/joint4_dctv2_rgb_scalefree.yaml
for d in hoi4d_processed_2d_v2 epic_processed_2d arctic_processed_2d sf3d_processed_v3; do
  mkdir -p /dev/shm/$d/data.lmdb; [ -f /dev/shm/$d/data.lmdb/data.mdb ] || cp /workspace/datasets/$d/data.lmdb/data.mdb /dev/shm/$d/data.lmdb/
done
for d in hoi4d_processed_2d_v2 epic_processed_2d arctic_processed_2d; do
  mkdir -p /dev/shm/$d/frames.lmdb; [ -f /dev/shm/$d/frames.lmdb/data.mdb ] || cp /workspace/datasets/$d/frames.lmdb/data.mdb /dev/shm/$d/frames.lmdb/
done
mkdir -p /dev/shm/sf3d_frames_512.lmdb; [ -f /dev/shm/sf3d_frames_512.lmdb/data.mdb ] || cp /workspace/datasets/sf3d_frames_512.lmdb/data.mdb /dev/shm/sf3d_frames_512.lmdb/
sed 's#/workspace/datasets/\(hoi4d_processed_2d_v2\|epic_processed_2d\|arctic_processed_2d\|sf3d_processed_v3\|sf3d_frames_512.lmdb\)#/dev/shm/\1#g' $CFG > /dev/shm/joint4v2_local.yaml
echo "staged paths: $(grep -c '/dev/shm/' /dev/shm/joint4v2_local.yaml)"; df -h /dev/shm | tail -1
mkdir -p experiments/$E/logs experiments/$E/checkpoints
echo "=== START $E $(date)"
HF_HOME=/root/hfcache HF_HUB_OFFLINE=1 /opt/venv/bin/python train_multi_better.py fit --config /dev/shm/joint4v2_local.yaml > experiments/$E/logs/train.log 2>&1
echo "=== END $E exit=$? $(date)"
best=$(ls experiments/$E/checkpoints/ | grep best- | sed 's/.*sf3dval\([0-9.]*\)\.ckpt/\1 &/' | sort -g | head -1 | cut -d' ' -f2)
if [ -n "$best" ]; then for f in experiments/$E/checkpoints/*.ckpt; do [ "$(basename "$f")" = "$best" ] || rm -f "$f"; done; echo "best=$best"; fi
echo "JOINT_TRAIN_DONE $(date)"
ck=$(ls experiments/$E/checkpoints/best-*.ckpt | head -1); [ -n "$ck" ] || { echo "NO_CKPT"; exit 1; }
# SF3D test with the scratch DCT config (identical model_params, 3D metrics), staged paths
mkdir -p /dev/shm/data.lmdb /dev/shm/frames.lmdb
[ -f /dev/shm/data.lmdb/data.mdb ] || ln -s /dev/shm/sf3d_processed_v3/data.lmdb/data.mdb /dev/shm/data.lmdb/data.mdb
[ -f /dev/shm/frames.lmdb/data.mdb ] || ln -s /dev/shm/sf3d_frames_512.lmdb/data.mdb /dev/shm/frames.lmdb/data.mdb
T() { logn=$1; shift; HF_HOME=/root/hfcache HF_HUB_OFFLINE=1 /opt/venv/bin/python train_SF3D_better.py test --config config/sf3d_train_runpod_g19_dct_rgb_scalefree.yaml --ckpt_path $ck --trainer.logger=false --model.model_params.trajectory_dct_pin_start true --model.model_params.trajectory_dct_scale_split true --model.model_params.compile_model false --data.lmdb_path /dev/shm/data.lmdb --data.frame_cache_path /dev/shm/frames.lmdb --data.num_workers_val 8 "$@" > experiments/$E/logs/$logn 2>&1; echo "=== TESTDONE $logn exit=$?"; }
T test_sf3d.log
T test_sf3d_gt_z0.log --model.config.test_trajectory_scale gt_z0
for c in hoi4d_v2 epic_v1 arctic_v1; do
  HF_HOME=/root/hfcache HF_HUB_OFFLINE=1 /opt/venv/bin/python train_SF3D_better.py test --config config/${c}_rgb_scalefree.yaml --ckpt_path $ck --trainer.logger=false --model.model_params.trajectory_dct_coeffs 6 --model.model_params.trajectory_dct_pin_start true --model.model_params.trajectory_dct_scale_split true --model.model_params.compile_model false --data.num_workers_val 8 > experiments/$E/logs/test_${c}.log 2>&1
  echo "=== TESTDONE $c exit=$?"
done
echo "JOINT_TEST_DONE $(date)"; echo "CHAIN_DONE $(date)"
