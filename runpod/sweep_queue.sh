#!/usr/bin/env bash
# Run several training arms back-to-back on ONE training pod (same
# staging/env as train_pod.sh launch, minus the per-launch ssh round trips).
# Runs ON THE POD, detached:
#   ssh segaff-<name> "( nohup bash /workspace/SegAffordance/runpod/sweep_queue.sh \
#       <exp_id>:<config> [<exp_id>:<config> ...] > /workspace/SegAffordance/sweep_queue.log 2>&1 < /dev/null & )"
# Each arm's own log: experiments/<exp_id>/logs/train.log. Skips an arm whose
# checkpoints/last.ckpt already exists (resume-safe across pod swaps).
set -u
cd /workspace/SegAffordance
bash runpod/ensure_env.sh
cat /workspace/models/RN50.pt > /dev/null 2>&1 || true
ulimit -n 65536
for arm in "$@"; do
  exp="${arm%%:*}"; cfg="${arm#*:}"
  if [ -f "experiments/${exp}/checkpoints/last.ckpt" ]; then
    echo "=== SKIP ${exp} (last.ckpt exists) $(date)"; continue
  fi
  droot=$(grep -E '^\s*train_data_dir:' "$cfg" | head -1 | sed 's/.*"\(.*\)".*/\1/')
  fcache=$(grep -E '^\s*frame_cache_path:' "$cfg" | head -1 | sed 's/.*"\(.*\)".*/\1/')
  [ -z "$fcache" ] && fcache="${droot}/frames.lmdb"
  # stage the LMDBs this config names into /dev/shm (re-stage if the arm
  # points elsewhere than the previous one)
  if [ ! -f /dev/shm/data.lmdb/data.mdb ] || [ "$(cat /dev/shm/.staged 2>/dev/null)" != "$droot|$fcache" ]; then
    rm -rf /dev/shm/data.lmdb /dev/shm/frames.lmdb
    mkdir -p /dev/shm/data.lmdb /dev/shm/frames.lmdb
    cp "${droot}/data.lmdb/data.mdb" /dev/shm/data.lmdb/
    cp "${fcache}/data.mdb" /dev/shm/frames.lmdb/
    echo "$droot|$fcache" > /dev/shm/.staged
  fi
  mkdir -p "experiments/${exp}/logs" "experiments/${exp}/checkpoints"
  echo "=== START ${exp} ${cfg} $(date)"
  HF_HOME=/root/hfcache HF_HUB_OFFLINE=1 /opt/venv/bin/python train_SF3D_better.py fit \
    --config "$cfg" --data.lmdb_path /dev/shm/data.lmdb --data.frame_cache_path /dev/shm/frames.lmdb \
    > "experiments/${exp}/logs/train.log" 2>&1
  echo "=== END ${exp} exit=$? $(date)"
  ls "experiments/${exp}/checkpoints/" | grep best- | sed 's/.*valloss\([0-9.]*\)\.ckpt/\1 &/' | sort -g | head -1
done
echo "QUEUE_DONE $(date)"
