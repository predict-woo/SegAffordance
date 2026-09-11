#!/bin/bash
# Pod jdec-attnpool (2026-09-13): joint decoder READOUT arm attnpool on the l2anchor recipe. Stage SF3D + the
# three hand LMDBs into /dev/shm, train with a pod-local config copy, keep the best ckpt, test SF3D
# (decoder length = head, and = writer constants for comparability) and each hand source with the
# single-source configs + decoder overrides. Markers: JOINT_TRAIN_DONE, JOINT_TEST_DONE, CHAIN_DONE.
cd /workspace/SegAffordance
bash runpod/ensure_env.sh
export TMPDIR=/workspace/tmp; mkdir -p $TMPDIR   # Lightning's checkpoint tempfile must not land on the pod overlay
export TORCHINDUCTOR_CACHE_DIR=/root/inductor_cache TRITON_CACHE_DIR=/root/triton_cache   # compile caches stay POD-LOCAL: under $TMPDIR on the network volume, concurrent pods hit "Stale file handle" (2026-09-13)
cat /workspace/cache/dinov3/*.pth > /dev/null 2>&1 || true
ulimit -n 65536
E=20260913_joint4_decoder_l2anchor_attnpool; CFG=config/joint4_decoder_l2anchor_attnpool.yaml
for d in hoi4d_processed_2d_v2 epic_processed_2d arctic_processed_2d sf3d_processed_v3; do
  mkdir -p /dev/shm/$d/data.lmdb; [ -f /dev/shm/$d/data.lmdb/data.mdb ] || cp /workspace/datasets/$d/data.lmdb/data.mdb /dev/shm/$d/data.lmdb/
done
for d in hoi4d_processed_2d_v2 epic_processed_2d arctic_processed_2d; do
  mkdir -p /dev/shm/$d/frames.lmdb; [ -f /dev/shm/$d/frames.lmdb/data.mdb ] || cp /workspace/datasets/$d/frames.lmdb/data.mdb /dev/shm/$d/frames.lmdb/
done
mkdir -p /dev/shm/sf3d_frames_512.lmdb; [ -f /dev/shm/sf3d_frames_512.lmdb/data.mdb ] || cp /workspace/datasets/sf3d_frames_512.lmdb/data.mdb /dev/shm/sf3d_frames_512.lmdb/
sed 's#/workspace/datasets/\(hoi4d_processed_2d_v2\|epic_processed_2d\|arctic_processed_2d\|sf3d_processed_v3\|sf3d_frames_512.lmdb\)#/dev/shm/\1#g' $CFG > /dev/shm/joint4dec_l2anchor_attnpool_local.yaml
echo "staged paths: $(grep -c '/dev/shm/' /dev/shm/joint4dec_l2anchor_attnpool_local.yaml)"; df -h /dev/shm | tail -1
mkdir -p experiments/$E/logs experiments/$E/checkpoints
echo "=== START $E $(date)"
HF_HOME=/root/hfcache HF_HUB_OFFLINE=1 /opt/venv/bin/python train_multi_better.py fit --config /dev/shm/joint4dec_l2anchor_attnpool_local.yaml > experiments/$E/logs/train.log 2>&1
echo "=== END $E exit=$? $(date)"
best=$(ls experiments/$E/checkpoints/ | grep best- | sed 's/.*sf3dval\([0-9.]*\)\.ckpt/\1 &/' | sort -g | head -1 | cut -d' ' -f2)
if [ -n "$best" ]; then for f in experiments/$E/checkpoints/*.ckpt; do [ "$(basename "$f")" = "$best" ] || rm -f "$f"; done; echo "best=$best"; fi
echo "JOINT_TRAIN_DONE $(date)"
ck=$(ls experiments/$E/checkpoints/best-*.ckpt | head -1); [ -n "$ck" ] || { echo "NO_CKPT"; exit 1; }
mkdir -p /dev/shm/data.lmdb /dev/shm/frames.lmdb
[ -f /dev/shm/data.lmdb/data.mdb ] || ln -s /dev/shm/sf3d_processed_v3/data.lmdb/data.mdb /dev/shm/data.lmdb/data.mdb
[ -f /dev/shm/frames.lmdb/data.mdb ] || ln -s /dev/shm/sf3d_frames_512.lmdb/data.mdb /dev/shm/frames.lmdb/data.mdb
T() { logn=$1; shift; HF_HOME=/root/hfcache HF_HUB_OFFLINE=1 /opt/venv/bin/python train_SF3D_better.py test --config config/sf3d_test_decoder_rgb_scalefree.yaml --ckpt_path $ck --trainer.logger=false --model.model_params.articulation_readout attnpool --model.model_params.readout_mask_eps 0.01 --model.model_params.compile_model false --data.lmdb_path /dev/shm/data.lmdb --data.frame_cache_path /dev/shm/frames.lmdb --data.num_workers_val 8 "$@" > experiments/$E/logs/$logn 2>&1; echo "=== TESTDONE $logn exit=$?"; }
T test_sf3d.log
T test_sf3d_writerlen.log --model.model_params.trajectory_decoder_length writer
DEC="--model.model_params.articulation_readout attnpool --model.model_params.readout_mask_eps 0.01 --model.model_params.use_trajectory_head false --model.model_params.trajectory_decoder analytic --model.model_params.trajectory_dct_coeffs 0"
for c in hoi4d_v2 epic_v1 arctic_v1; do
  HF_HOME=/root/hfcache HF_HUB_OFFLINE=1 /opt/venv/bin/python train_SF3D_better.py test --config config/${c}_rgb_scalefree.yaml --ckpt_path $ck --trainer.logger=false $DEC --model.model_params.compile_model false --data.num_workers_val 8 > experiments/$E/logs/test_${c}.log 2>&1
  echo "=== TESTDONE $c exit=$?"
done
echo "JOINT_TEST_DONE $(date)"; echo "CHAIN_DONE $(date)"
