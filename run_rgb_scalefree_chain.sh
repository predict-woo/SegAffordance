#!/bin/bash
# Pod A chain (2026-09-09): HOI4D rgb_scalefree arm -> its test -> resolve the
# best ckpt into the SF3D ft config -> SF3D post-training -> test (pred_z_p)
# -> test (gt_z0 oracle scale). Markers: HOI4D_ARM_DONE, SF3D_FT_DONE, CHAIN_DONE.
cd /workspace/SegAffordance
T() { # exp logname extra-args...
  exp=experiments/$1; logn=$2; shift 2; ck=$(ls $exp/checkpoints/best-*.ckpt | head -1)
  echo "=== TEST $exp -> $logn $ck $(date)"
  HF_HOME=/root/hfcache HF_HUB_OFFLINE=1 /opt/venv/bin/python train_SF3D_better.py test --config $exp/config.yaml --ckpt_path $ck --trainer.logger=false --data.lmdb_path /dev/shm/data.lmdb --data.frame_cache_path /dev/shm/frames.lmdb --data.num_workers_val 8 "$@" > $exp/logs/$logn 2>&1
  echo "=== TESTDONE $exp $logn exit=$?"
}
H=20260909_hoi4d_2d_v2_rgb_scalefree
bash runpod/sweep_queue.sh $H:config/hoi4d_v2_rgb_scalefree.yaml
T $H test.log
echo "HOI4D_ARM_DONE $(date)"
ck=$(ls experiments/$H/checkpoints/best-*.ckpt | head -1)
[ -n "$ck" ] || { echo "NO_HOI4D_CKPT"; exit 1; }
sed -i "s#finetune_from_path: \".*\"#finetune_from_path: \"/workspace/SegAffordance/$ck\"#" config/sf3d_train_runpod_g19_dct_rgb_scalefree_ft_hoi4d.yaml experiments/20260909_sf3d_g19_dct_rgb_scalefree_ft_hoi4d/config.yaml
grep finetune_from_path config/sf3d_train_runpod_g19_dct_rgb_scalefree_ft_hoi4d.yaml
F=20260909_sf3d_g19_dct_rgb_scalefree_ft_hoi4d
bash runpod/sweep_queue.sh $F:config/sf3d_train_runpod_g19_dct_rgb_scalefree_ft_hoi4d.yaml
T $F test.log
T $F test_gt_z0.log --model.config.test_trajectory_scale gt_z0
echo "SF3D_FT_DONE $(date)"
echo "CHAIN_DONE $(date)"
