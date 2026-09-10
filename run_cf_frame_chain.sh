#!/bin/bash
# Pod cfframe (2026-09-11): the shape-designed closed-form loss arm (same pipeline as the
# August cf arms: sweep_queue stages SF3D into /dev/shm, trains, keeps the best ckpt) -> test.
# Markers: QUEUE_DONE (from sweep_queue), CF_FRAME_DONE, CHAIN_DONE.
cd /workspace/SegAffordance
bash runpod/ensure_env.sh
export TMPDIR=/workspace/tmp; mkdir -p $TMPDIR   # Lightning's fsspec transaction tempfile must not land on the pod overlay
E=20260911_sf3d_cf_frame; CFG=config/sf3d_train_runpod_cf_frame.yaml
bash runpod/sweep_queue.sh $E:$CFG
ck=$(ls experiments/$E/checkpoints/best-*.ckpt | head -1); [ -n "$ck" ] || { echo "NO_CKPT"; exit 1; }
echo "=== TEST $E $ck $(date)"
HF_HOME=/root/hfcache HF_HUB_OFFLINE=1 /opt/venv/bin/python train_SF3D_better.py test --config experiments/$E/config.yaml --ckpt_path $ck --trainer.logger=false --data.lmdb_path /dev/shm/data.lmdb --data.frame_cache_path /dev/shm/frames.lmdb --data.num_workers_val 8 > experiments/$E/logs/test.log 2>&1
echo "=== TESTDONE $E exit=$?"
echo "CF_FRAME_DONE $(date)"; echo "CHAIN_DONE $(date)"
