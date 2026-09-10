#!/bin/bash
# Pod M (2026-09-10 night): multi3 DCT v2 2D arm (pinned start + scale split + uv fdiff trio + log length) (12 ep) -> union test + per-source tests -> SF3D DCT
# post-training from its best ckpt (sweep_queue stages SF3D) -> tests pred_z_p / gt_z0.
# Markers: MULTI3DCT_TRAIN_DONE, MULTI3DCT_TEST_DONE, POST_START, SF3D_DCT_DONE, CHAIN_DONE.
cd /workspace/SegAffordance
bash runpod/ensure_env.sh
# Lightning saves checkpoints via an fsspec transaction = a tempfile in $TMPDIR then a move;
# the pod overlay's /tmp can be smaller than a 3.7 GB ckpt (ENOSPC, dev pod 2026-09-10). Keep it on the volume.
export TMPDIR=/workspace/tmp; mkdir -p $TMPDIR
cat /workspace/cache/dinov3/*.pth > /dev/null 2>&1 || true
ulimit -n 65536
E=20260910_multi3_dctv2_rgb_scalefree; CFG=config/multi3_dctv2_rgb_scalefree.yaml
for d in hoi4d_processed_2d_v2 epic_processed_2d arctic_processed_2d; do
  mkdir -p /dev/shm/$d/data.lmdb /dev/shm/$d/frames.lmdb
  [ -f /dev/shm/$d/data.lmdb/data.mdb ] || cp /workspace/datasets/$d/data.lmdb/data.mdb /dev/shm/$d/data.lmdb/
  [ -f /dev/shm/$d/frames.lmdb/data.mdb ] || cp /workspace/datasets/$d/frames.lmdb/data.mdb /dev/shm/$d/frames.lmdb/
done
sed 's#/workspace/datasets/\(hoi4d_processed_2d_v2\|epic_processed_2d\|arctic_processed_2d\)#/dev/shm/\1#g' $CFG > /dev/shm/multi3dctv2_local.yaml
echo "staged paths: $(grep -c '/dev/shm/' /dev/shm/multi3dctv2_local.yaml)"
mkdir -p experiments/$E/logs experiments/$E/checkpoints
echo "=== START $E $(date)"
HF_HOME=/root/hfcache HF_HUB_OFFLINE=1 /opt/venv/bin/python train_multi_better.py fit --config /dev/shm/multi3dctv2_local.yaml > experiments/$E/logs/train.log 2>&1
echo "=== END $E exit=$? $(date)"
best=$(ls experiments/$E/checkpoints/ | grep best- | sed 's/.*valloss\([0-9.]*\)\.ckpt/\1 &/' | sort -g | head -1 | cut -d' ' -f2)
if [ -n "$best" ]; then for f in experiments/$E/checkpoints/*.ckpt; do [ "$(basename "$f")" = "$best" ] || rm -f "$f"; done; echo "best=$best"; fi
echo "MULTI3DCT_TRAIN_DONE $(date)"
ck=$(ls experiments/$E/checkpoints/best-*.ckpt | head -1); [ -n "$ck" ] || { echo "NO_CKPT"; exit 1; }
HF_HOME=/root/hfcache HF_HUB_OFFLINE=1 /opt/venv/bin/python train_multi_better.py test --config /dev/shm/multi3dctv2_local.yaml --ckpt_path $ck --trainer.logger=false --data.num_workers_val 8 > experiments/$E/logs/test.log 2>&1
echo "=== TESTDONE union exit=$?"
for c in hoi4d_v2 epic_v1 arctic_v1; do
  HF_HOME=/root/hfcache HF_HUB_OFFLINE=1 /opt/venv/bin/python train_SF3D_better.py test --config config/${c}_rgb_scalefree.yaml --ckpt_path $ck --trainer.logger=false --model.model_params.trajectory_dct_coeffs 6 --model.model_params.trajectory_dct_pin_start true --model.model_params.trajectory_dct_scale_split true --model.model_params.compile_model false --data.num_workers_val 8 > experiments/$E/logs/test_${c}.log 2>&1
  echo "=== TESTDONE $c exit=$?"
done
echo "MULTI3DCT_TEST_DONE $(date)"
F=20260910_sf3d_g19_dctv2_rgb_scalefree_ft_multi3dctv2; FCFG=config/sf3d_train_runpod_g19_dctv2_rgb_scalefree_ft_multi3dctv2.yaml
sed -i "s#finetune_from_path: \".*\"#finetune_from_path: \"/workspace/SegAffordance/$ck\"#" $FCFG experiments/$F/config.yaml
grep finetune_from_path $FCFG
rm -rf /dev/shm/hoi4d_processed_2d_v2 /dev/shm/epic_processed_2d /dev/shm/arctic_processed_2d /dev/shm/multi3dctv2_local.yaml
echo "POST_START $(date)"
bash runpod/sweep_queue.sh $F:$FCFG
T() { exp=experiments/$1; logn=$2; shift 2; ck=$(ls $exp/checkpoints/best-*.ckpt | head -1)
  echo "=== TEST $exp -> $logn $ck $(date)"
  HF_HOME=/root/hfcache HF_HUB_OFFLINE=1 /opt/venv/bin/python train_SF3D_better.py test --config $exp/config.yaml --ckpt_path $ck --trainer.logger=false --data.lmdb_path /dev/shm/data.lmdb --data.frame_cache_path /dev/shm/frames.lmdb --data.num_workers_val 8 "$@" > $exp/logs/$logn 2>&1
  echo "=== TESTDONE $exp $logn exit=$?"; }
T $F test.log
T $F test_gt_z0.log --model.config.test_trajectory_scale gt_z0
echo "SF3D_DCT_DONE $(date)"; echo "CHAIN_DONE $(date)"
