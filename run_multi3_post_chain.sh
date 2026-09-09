#!/bin/bash
# Pod C phase 2 (2026-09-09): wait for the multi3 arm's test, resolve its best
# ckpt into the PLAIN-head SF3D post-training config, train (sweep_queue stages
# the SF3D LMDBs into /dev/shm), test with pred_z_p and gt_z0.
# Markers: POST_WAITING, POST_START, SF3D_PLAIN_DONE, CHAIN2_DONE.
cd /workspace/SegAffordance
echo "POST_WAITING $(date)"
until grep -q "^MULTI3_TEST_DONE" multi3_chain.log 2>/dev/null; do sleep 60; done
M=20260909_multi3_rgb_scalefree; ck=$(ls experiments/$M/checkpoints/best-*.ckpt 2>/dev/null | head -1)
[ -n "$ck" ] || { echo "NO_MULTI3_CKPT"; exit 1; }
F=20260909_sf3d_plain_rgb_scalefree_ft_multi3; CFG=config/sf3d_train_runpod_plain_rgb_scalefree_ft_multi3.yaml
sed -i "s#finetune_from_path: \".*\"#finetune_from_path: \"/workspace/SegAffordance/$ck\"#" $CFG experiments/$F/config.yaml
grep finetune_from_path $CFG
# free the 2D staging (the SF3D LMDBs are ~26 GB) and let sweep_queue stage SF3D
rm -rf /dev/shm/hoi4d_processed_2d_v2 /dev/shm/epic_processed_2d /dev/shm/arctic_processed_2d /dev/shm/multi3_local.yaml
echo "POST_START $(date)"
bash runpod/sweep_queue.sh $F:$CFG
T() { exp=experiments/$1; logn=$2; shift 2; ck=$(ls $exp/checkpoints/best-*.ckpt | head -1)
  echo "=== TEST $exp -> $logn $ck $(date)"
  HF_HOME=/root/hfcache HF_HUB_OFFLINE=1 /opt/venv/bin/python train_SF3D_better.py test --config $exp/config.yaml --ckpt_path $ck --trainer.logger=false --data.lmdb_path /dev/shm/data.lmdb --data.frame_cache_path /dev/shm/frames.lmdb --data.num_workers_val 8 "$@" > $exp/logs/$logn 2>&1
  echo "=== TESTDONE $exp $logn exit=$?"; }
T $F test.log
T $F test_gt_z0.log --model.config.test_trajectory_scale gt_z0
echo "SF3D_PLAIN_DONE $(date)"; echo "CHAIN2_DONE $(date)"
