#!/bin/bash
# Pod B (2026-09-09): SF3D rgb_scalefree SCRATCH arm -> test (pred_z_p) -> test (gt_z0).
cd /workspace/SegAffordance
T() { exp=experiments/$1; logn=$2; shift 2; ck=$(ls $exp/checkpoints/best-*.ckpt | head -1)
  echo "=== TEST $exp -> $logn $ck $(date)"
  HF_HOME=/root/hfcache HF_HUB_OFFLINE=1 /opt/venv/bin/python train_SF3D_better.py test --config $exp/config.yaml --ckpt_path $ck --trainer.logger=false --data.lmdb_path /dev/shm/data.lmdb --data.frame_cache_path /dev/shm/frames.lmdb --data.num_workers_val 8 "$@" > $exp/logs/$logn 2>&1
  echo "=== TESTDONE $exp $logn exit=$?"; }
S=20260909_sf3d_g19_dct_rgb_scalefree
bash runpod/sweep_queue.sh $S:config/sf3d_train_runpod_g19_dct_rgb_scalefree.yaml
T $S test.log
T $S test_gt_z0.log --model.config.test_trajectory_scale gt_z0
echo "SCRATCH_DONE $(date)"; echo "CHAIN_DONE $(date)"
