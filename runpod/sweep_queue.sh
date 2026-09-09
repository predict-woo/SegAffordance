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
  # Multi-source configs (config/multi*.yaml): several LMDBs, read from the
  # volume directly (the hand datasets are small) via train_multi_better.py;
  # no /dev/shm staging and no lmdb_path overrides.
  script=train_SF3D_better.py; extra="--data.lmdb_path /dev/shm/data.lmdb --data.frame_cache_path /dev/shm/frames.lmdb"
  case "$(basename "$cfg")" in multi*) script=train_multi_better.py; extra="";; esac
  droot=$(grep -E '^\s*train_data_dir:' "$cfg" | head -1 | sed 's/.*"\(.*\)".*/\1/')
  fcache=$(grep -E '^\s*frame_cache_path:' "$cfg" | head -1 | sed 's/.*"\(.*\)".*/\1/')
  [ -z "$fcache" ] && fcache="${droot}/frames.lmdb"
  # stage the LMDBs this config names into /dev/shm (re-stage if the arm
  # points elsewhere than the previous one)
  if [ -n "$extra" ] && { [ ! -f /dev/shm/data.lmdb/data.mdb ] || [ "$(cat /dev/shm/.staged 2>/dev/null)" != "$droot|$fcache" ]; }; then
    rm -rf /dev/shm/data.lmdb /dev/shm/frames.lmdb
    mkdir -p /dev/shm/data.lmdb /dev/shm/frames.lmdb
    cp "${droot}/data.lmdb/data.mdb" /dev/shm/data.lmdb/
    cp "${fcache}/data.mdb" /dev/shm/frames.lmdb/
    echo "$droot|$fcache" > /dev/shm/.staged
  fi
  mkdir -p "experiments/${exp}/logs" "experiments/${exp}/checkpoints"
  echo "=== START ${exp} ${cfg} $(date)"
  HF_HOME=/root/hfcache HF_HUB_OFFLINE=1 /opt/venv/bin/python $script fit \
    --config "$cfg" $extra \
    > "experiments/${exp}/logs/train.log" 2>&1
  echo "=== END ${exp} exit=$? $(date)"
  # Keep ONLY the best checkpoint: 4.3 GB per file, and 7 arms x (3 best +
  # last) blew the 1 TB volume quota on 2026-09-06 (both pods' jobs died
  # silently at "Disk quota exceeded").
  best=$(ls "experiments/${exp}/checkpoints/" | grep best- | sed 's/.*valloss\([0-9.]*\)\.ckpt/\1 &/' | sort -g | head -1 | cut -d' ' -f2)
  if [ -n "$best" ]; then
    for f in experiments/${exp}/checkpoints/*.ckpt; do
      [ "$(basename "$f")" = "$best" ] || rm -f "$f"
    done
    echo "$best"
  fi
done
echo "QUEUE_DONE $(date)"
