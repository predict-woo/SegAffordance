#!/bin/bash
# Pod C (2026-09-09): stage the three 2D LMDBs into /dev/shm, train the multi3
# rgb_scalefree arm with a pod-local config copy pointing at /dev/shm, keep the
# best ckpt only, test on the val union. Markers: MULTI3_TRAIN_DONE, MULTI3_TEST_DONE, CHAIN_DONE.
cd /workspace/SegAffordance
bash runpod/ensure_env.sh
cat /workspace/cache/dinov3/*.pth > /dev/null 2>&1 || true
ulimit -n 65536
E=20260909_multi3_rgb_scalefree; CFG=config/multi3_rgb_scalefree.yaml
for d in hoi4d_processed_2d_v2 epic_processed_2d arctic_processed_2d; do
  mkdir -p /dev/shm/$d/data.lmdb /dev/shm/$d/frames.lmdb
  [ -f /dev/shm/$d/data.lmdb/data.mdb ] || cp /workspace/datasets/$d/data.lmdb/data.mdb /dev/shm/$d/data.lmdb/
  [ -f /dev/shm/$d/frames.lmdb/data.mdb ] || cp /workspace/datasets/$d/frames.lmdb/data.mdb /dev/shm/$d/frames.lmdb/
done
sed 's#/workspace/datasets/\(hoi4d_processed_2d_v2\|epic_processed_2d\|arctic_processed_2d\)/#/dev/shm/\1/#g' $CFG > /dev/shm/multi3_local.yaml
grep -c "/dev/shm/" /dev/shm/multi3_local.yaml
mkdir -p experiments/$E/logs experiments/$E/checkpoints
echo "=== START $E $(date)"
HF_HOME=/root/hfcache HF_HUB_OFFLINE=1 /opt/venv/bin/python train_multi_better.py fit --config /dev/shm/multi3_local.yaml > experiments/$E/logs/train.log 2>&1
echo "=== END $E exit=$? $(date)"
best=$(ls experiments/$E/checkpoints/ | grep best- | sed 's/.*valloss\([0-9.]*\)\.ckpt/\1 &/' | sort -g | head -1 | cut -d' ' -f2)
if [ -n "$best" ]; then for f in experiments/$E/checkpoints/*.ckpt; do [ "$(basename "$f")" = "$best" ] || rm -f "$f"; done; echo "best=$best"; fi
echo "MULTI3_TRAIN_DONE $(date)"
ck=$(ls experiments/$E/checkpoints/best-*.ckpt | head -1)
[ -n "$ck" ] || { echo "NO_CKPT"; exit 1; }
echo "=== TEST $E $ck $(date)"
HF_HOME=/root/hfcache HF_HUB_OFFLINE=1 /opt/venv/bin/python train_multi_better.py test --config /dev/shm/multi3_local.yaml --ckpt_path $ck --trainer.logger=false --data.num_workers_val 8 > experiments/$E/logs/test.log 2>&1
echo "=== TESTDONE exit=$?"
echo "MULTI3_TEST_DONE $(date)"; echo "CHAIN_DONE $(date)"
